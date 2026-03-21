from flask import Flask, jsonify, request, render_template, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from data_engine import DestinationDatabase 
import os
import csv
from datetime import datetime
from dotenv import load_dotenv # Import dotenv
import gspread
import json

# Load environment variables from the .env file
load_dotenv()

# --- GOOGLE SHEETS SETUP ---
try:
    # Authenticate using the downloaded JSON file
    gc = gspread.service_account(filename='google_secrets.json')
    SHEET_ID = os.getenv('SHEET_ID')
    worksheet = gc.open_by_key(SHEET_ID).sheet1
except Exception as e:
    print(f"Failed to connect to Google Sheets: {e}")

app = Flask(__name__)

# Fetch the secret key from the .env file (with a fallback just in case)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_fallback_key')

db = DestinationDatabase()
df, similarity_matrix = db.get_data()

# SECURE ADMIN CREDENTIALS
# Fetch from .env and hash the password
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'superadmin')
raw_admin_password = os.getenv('ADMIN_PASSWORD', 'default_password')
ADMIN_PASSWORD_HASH = generate_password_hash(raw_admin_password)

# RECOMMENDATION ALGORITHM FUNCTIONS
def generate_explanation(row, rel_score, tags, reason, month, persona, density):
    match_pct = int(rel_score * 100)
    if reason == "Top Match": return f"A {match_pct}% match for your {persona} profile."
    elif reason == "Critique Match": return f"We found this similar destination that better fits your request."
    else: 
        category = str(row.get('Category', 'destination')).lower()
        time_context = f"in {month}" if month else "annually"
        sustainable_threshold = 250 if month else 3000
        if density < sustainable_threshold: 
            return f"Sustainable Alternative: This {category} is an excellent match, but offers a highly sustainable tourist density ({density:,} per km² {time_context}) compared to crowded hotspots."
        else:
            return f"Exploration Pick: We included this {category} to diversify your trip. It contrasts with your top matches to offer a different regional experience."

def calculate_mmr(relevance, sim_matrix, popularity, lambda_param, valid_indices, alpha=0.5, beta=0.5, top_k=6):
    remaining_candidates = list(valid_indices) 
    selected_list = []
    if not remaining_candidates: return []
    first_item = max(remaining_candidates, key=lambda i: relevance[i])
    selected_list.append(first_item)
    remaining_candidates.remove(first_item)
    while len(selected_list) < top_k and remaining_candidates:
        mmr_scores = {}
        for candidate in remaining_candidates:
            rel = relevance[candidate]
            sim_penalty = max([sim_matrix[candidate][j] for j in selected_list])
            pop_penalty = popularity[candidate] 
            score = (lambda_param * rel) - ((1 - lambda_param) * ((alpha * sim_penalty) + (beta * pop_penalty)))
            mmr_scores[candidate] = score
        best_next = max(mmr_scores, key=mmr_scores.get)
        selected_list.append(best_next)
        remaining_candidates.remove(best_next)
    return selected_list

def calculate_ild(selected_indices, sim_matrix):
    k = len(selected_indices)
    if k <= 1: return 0.0
    total_distance = 0
    for i in selected_indices:
        for j in selected_indices:
            if i != j: total_distance += (1 - sim_matrix[i][j])
    return total_distance / (k * (k - 1))

# --- MAIN ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/log', methods=['POST'])
def log_action():
    try:
        data = request.json
        if data.get('is_dev'): return jsonify({'status': 'ignored (developer mode)'})
            
        # Append row directly to Google Sheets
        row_data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get('session_id', 'unknown'),
            data.get('action', 'unknown_action'),
            data.get('item', ''),           
            data.get('value', ''),          
            data.get('lambda_val', ''),
            data.get('persona', ''),
            data.get('month', ''),
            data.get('tags', ''),
            data.get('critique_target', ''),
            data.get('critique_type', '')
        ]
        
        worksheet.append_row(row_data)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    try:
        lambda_param = float(request.args.get('lambda', 0.5))
        tags_param = request.args.get('tags', '')
        tags = [t for t in tags_param.split(',') if t]
        month = request.args.get('month', '') 
        persona = request.args.get('persona', 'Standard')
        critique_target = request.args.get('critique_target', '')
        critique_type = request.args.get('critique_type', '')
        avoid_overtourism = request.args.get('avoid_overtourism', 'false') == 'true'
        top_k = 6
        
        if critique_target: dynamic_relevance = db.get_critique_relevance(critique_target)
        else: dynamic_relevance = db.get_semantic_relevance(tags, persona)

        raw_densities = []
        for _, row in df.iterrows():
            region = str(row['Region'])
            seasonal_mult_regional = db.get_seasonal_multiplier(month, region) if month else 1.0
            adj_traffic = row.get('Clean_Traffic', 0) * seasonal_mult_regional
            area = float(row.get('Area_sq_km', 100.0))
            time_traffic = (adj_traffic / 12) if month else adj_traffic
            density = time_traffic / area if area > 0 else 0
            raw_densities.append(density)
            
        max_density = max(raw_densities) if raw_densities and max(raw_densities) > 0 else 1
        popularity_list = [d / max_density for d in raw_densities]

        valid_indices = list(range(len(df)))
        overtourism_threshold = 250 if month else 3000
        
        # filter Avoid Overtourism
        if avoid_overtourism:
            valid_indices = [i for i in valid_indices if raw_densities[i] < overtourism_threshold]

        if critique_target and critique_type:
            target_idx = df[df['Destination'] == critique_target].index[0]
            target_density = raw_densities[target_idx]
            target_price = df.iloc[target_idx]['Daily_Budget_USD']
            if critique_type == 'quieter': valid_indices = [i for i in valid_indices if raw_densities[i] < target_density]
            elif critique_type == 'cheaper': valid_indices = [i for i in valid_indices if df.iloc[i]['Daily_Budget_USD'] < target_price]

        final_list = calculate_mmr(dynamic_relevance, similarity_matrix, popularity_list, lambda_param, valid_indices, alpha=0.2, beta=0.8, top_k=top_k)
        
        avg_accuracy = float(np.mean([dynamic_relevance[i] for i in final_list])) if final_list else 0.0
        ild_score = float(calculate_ild(final_list, similarity_matrix))
        
        recommended_items = []
        serendipity_scores = []
        prices = []
        
        for idx in final_list:
            row = df.iloc[idx]
            tourists_per_km2 = int(raw_densities[idx])
            is_overtourism = bool(tourists_per_km2 >= overtourism_threshold)
            if critique_target: reason = "Critique Match"
            else:
                top_matches = sorted(valid_indices, key=lambda i: dynamic_relevance[i], reverse=True)[:top_k]
                reason = "Top Match" if idx in top_matches else "Diversity Pick"
                
            pop = popularity_list[idx]
            rel = float(dynamic_relevance[idx])
            serendipity_scores.append((1 - pop) * rel)
            prices.append(float(row.get('Daily_Budget_USD', 100)))
            density_label = f"in {month}" if month else "annually"
            explanation_text = generate_explanation(row, rel, tags, reason, month, persona, tourists_per_km2)
            
            recommended_items.append({
                'name': str(row['Destination']),
                'country': str(row['Country']),
                'region': str(row['Region']),
                'category': str(row['Category']),
                'price': int(row.get('Daily_Budget_USD', 100)),
                'description': str(row.get('Description', 'A beautiful destination to explore.')),
                'overtourism': is_overtourism,
                'reason': reason,
                'xai_explanation': explanation_text,
                'tourist_density': tourists_per_km2,
                'density_label': density_label,
                'lat': float(row.get('Latitude', 0.0)),
                'lng': float(row.get('Longitude', 0.0))
            })
            
        return jsonify({
            'lambda': lambda_param,
            'accuracy': avg_accuracy,
            'diversity': ild_score,
            'serendipity': float(np.mean(serendipity_scores)) if serendipity_scores else 0.0,
            'avg_price': int(np.mean(prices)) if prices else 0,
            'items': recommended_items,
            'debug_info': {'persona': persona, 'month': month, 'critique': critique_type, 'avoid_overtourism': avoid_overtourism}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# SECURE ADMIN DASHBOARD & ADVANCED METADATA
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            error = "Invalid username or password. Access denied."
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('is_admin', None)
    return redirect(url_for('login'))

@app.route('/admin')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('login'))
    return render_template('admin.html')

@app.route('/api/admin/data')
def get_admin_data():
    if not session.get('is_admin'):
        return jsonify({'error': 'Unauthorized'}), 403

    stats = {
        'total_users': set(),
        'personas': {},
        'critiques': {'quieter': 0, 'cheaper': 0},
        'filters': {'avoid_overtourism': 0},
        'likes': {'TopMatch': 0, 'DiversityPick': 0, 'CritiqueMatch': 0},
        'dislikes': {'TopMatch': 0, 'DiversityPick': 0, 'CritiqueMatch': 0},
        'actions_breakdown': {},
        'final_bookings': {},
        'lambda_values': [],
        'comments': [] 
    }
    
    try:
        # Fetch all records from the Google Sheet
        # get_all_records() automatically uses Row 1 as dictionary keys
        records = worksheet.get_all_records()
        
        for row in records:
            # Convert to strings to avoid type errors from empty Google Sheet cells
            session_id = str(row.get('SessionID', ''))
            action = str(row.get('Action', ''))
            
            if session_id and session_id != 'unknown':
                stats['total_users'].add(session_id)
                
            if action == 'Completed_Onboarding':
                p = str(row.get('Persona', 'Unknown'))
                stats['personas'][p] = stats['personas'].get(p, 0) + 1
            
            if action:
                stats['actions_breakdown'][action] = stats['actions_breakdown'].get(action, 0) + 1
                
            if action == 'Clicked_Critique_quieter': stats['critiques']['quieter'] += 1
            if action == 'Clicked_Critique_cheaper': stats['critiques']['cheaper'] += 1
            
            if action == 'Toggled_Avoid_Overtourism' and str(row.get('Interaction_Value', '')).lower() == 'true':
                stats['filters']['avoid_overtourism'] += 1
            
            if row.get('Lambda_Value') != '':
                try: 
                    stats['lambda_values'].append(float(row.get('Lambda_Value')))
                except ValueError: 
                    pass
            
            if action.startswith('Rated_'):
                reason = action.replace('Rated_', '')
                val = str(row.get('Interaction_Value', ''))
                if val == 'Like': stats['likes'][reason] = stats['likes'].get(reason, 0) + 1
                elif val == 'Dislike': stats['dislikes'][reason] = stats['dislikes'].get(reason, 0) + 1
                
            if action == 'Selected_Final_Destination':
                dest = str(row.get('Item_Interacted', 'Unknown'))
                stats['final_bookings'][dest] = stats['final_bookings'].get(dest, 0) + 1
                
            if action == 'Submitted_Comment':
                stats['comments'].append({
                    'timestamp': str(row.get('Timestamp', '')),
                    'text': str(row.get('Interaction_Value', '')),
                    'session': session_id
                })

    except Exception as e:
        # If Google Sheets fails, return an error message instead of crashing
        return jsonify({'error': f'Failed to fetch data from Google Sheets: {str(e)}'}), 500

    # Post-processing stats
    stats['total_users'] = len(stats['total_users'])
    
    if stats['lambda_values']:
        stats['avg_lambda'] = round(sum(stats['lambda_values']) / len(stats['lambda_values']), 2)
    else:
        stats['avg_lambda'] = 0.5
        
    sorted_bookings = sorted(stats['final_bookings'].items(), key=lambda item: item[1], reverse=True)[:5]
    stats['top_bookings'] = dict(sorted_bookings)
    
    stats['comments'].reverse()
    
    return jsonify(stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)