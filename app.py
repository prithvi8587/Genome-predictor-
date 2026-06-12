import os
import joblib
import math
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file missing at {MODEL_PATH}")

# Standard Molecular Masses (g/mol)
MOL_WEIGHTS = {'A': 313.2, 'T': 304.2, 'C': 289.2, 'G': 329.2}

def compute_ultra_genomic_metrics(seq):
    if not seq:
        return {}
    
    length = len(seq)
    a, t, c, g = seq.count('A'), seq.count('T'), seq.count('C'), seq.count('G')
    
    # 1. Standard Percentages
    gc_content = ((g + c) / length) * 100
    at_content = ((a + t) / length) * 100
    
    # 2. Stranded Skew Analysis (Genomic Directionality indicators)
    gc_skew = (g - c) / (g + c) if (g + c) > 0 else 0.0
    at_skew = (a - t) / (a + t) if (a + t) > 0 else 0.0
    
    # 3. Structural Ring Mass Ratios
    purines = a + g
    pyrimidines = c + t
    
    # 4. Hydrogen Bonding Integrity (Thermal Stability proxy)
    h_bonds = (3 * (g + c)) + (2 * (a + t))
    
    # 5. Shannon Entropy Calculation
    entropy = 0.0
    for count in [a, t, c, g]:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)
            
    # 6. Homopolymer Slippage Check (Longest run of identical bases)
    max_run = 1
    current_run = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i-1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1

    return {
        "length": length,
        "gc_content": round(gc_content, 1),
        "at_content": round(at_content, 1),
        "gc_skew": round(gc_skew, 3),
        "at_skew": round(at_skew, 3),
        "purine_ratio": round((purines / length) * 100, 1),
        "pyrimidine_ratio": round((pyrimidines / length) * 100, 1),
        "hydrogen_bonds": h_bonds,
        "shannon_entropy": round(entropy, 3),
        "homopolymer_run": max_run,
        "molecular_mass": round(sum(MOL_WEIGHTS.get(base, 0) for base in seq), 1)
    }

def evaluate_structural_impact(ref, alt):
    ref_len, alt_len = len(ref), len(alt)
    diff = abs(ref_len - alt_len)
    
    if ref_len == 1 and alt_len == 1:
        is_transition = {ref, alt} in [{"A", "G"}, {"C", "T"}]
        mech = "Transition (Isomorphic Ring Swap)" if is_transition else "Transversion (Steric Hindrance Shift)"
        return mech, "Point Mutation (Potential Codon Alteration)"
    
    frame_status = "In-Frame Mutation (Preserved Triplet Phase)" if diff % 3 == 0 else "Frameshift Mutation (Disruptive Downstream Translation Run)"
    if ref_len > alt_len:
        return f"Nucleotide Deletion (-{diff} bp)", frame_status
    elif ref_len < alt_len:
        return f"Nucleotide Insertion (+{diff} bp)", frame_status
    return "Complex Segment Rearrangement", "Multi-base Substitution Block"

# --- ML Predict Route ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ref = data.get("ref", "").upper().strip()
        alt = data.get("alt", "").upper().strip()
        
        if not ref or not alt:
            return jsonify({"status": "error", "message": "Incomplete input alleles"}), 400
            
        ref_len, alt_len = len(ref), len(alt)
        var_type = 0 if (ref_len == 1 and alt_len == 1) else (1 if ref_len > alt_len else (2 if ref_len < alt_len else 3))
        
        # Random Forest Prediction
        prediction = int(model.predict([[ref_len, alt_len, var_type]])[0])
        confidence = float(model.predict_proba([[ref_len, alt_len, var_type]])[0][prediction]) * 100
        
        # Deep Metrics Compilation
        ref_metrics = compute_ultra_genomic_metrics(ref)
        alt_metrics = compute_ultra_genomic_metrics(alt)
        mechanism, consequence = evaluate_structural_impact(ref, alt)
        
        # Advanced Comparative Deltas
        mass_delta = round(alt_metrics["molecular_mass"] - ref_metrics["molecular_mass"], 1)
        bond_delta = alt_metrics["hydrogen_bonds"] - ref_metrics["hydrogen_bonds"]
        
        return jsonify({
            "status": "success",
            "prediction": "Pathogenic" if prediction == 1 else "Benign",
            "confidence": f"{confidence:.2f}%",
            "variant_class": ["Single Nucleotide Variant (SNV)", "Deletion Sequence", "Insertion Sequence", "Complex Block Indel"][var_type],
            "molecular_mechanism": mechanism,
            "predicted_consequence": consequence,
            "comparative_deltas": {
                "mass_shift_g_mol": f"{mass_delta:+.1f}",
                "hydrogen_bond_shift": f"{bond_delta:+d}"
            },
            "ref_profile": ref_metrics,
            "alt_profile": alt_metrics
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Secure Gemini Explanation Route ---
@app.route('/explain', methods=['POST'])
def explain():
    try:
        data = request.get_json()
        metrics = data.get("metrics", {})
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"insight": "Configuration missing: Please add GEMINI_API_KEY to Render environment variables."}), 200

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        prompt = f"""
        As an expert clinical bioinformatician, analyze this genetic variant data:
        - Prediction: {metrics.get('prediction')}
        - Confidence: {metrics.get('confidence')}
        - Class: {metrics.get('variant_class')}
        - Consequence: {metrics.get('predicted_consequence')}
        - Mass Shift: {metrics.get('comparative_deltas', {}).get('mass_shift_g_mol')} g/mol
        - Hydrogen Bond Shift: {metrics.get('comparative_deltas', {}).get('hydrogen_bond_shift')} bonds
        
        Provide a concise, 3-sentence scientific insight explaining what these changes mean for the DNA structure and protein translation. Keep it professional.
        """

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
        response_json = response.json()
        insight_text = response_json['candidates'][0]['content']['parts'][0]['text']
        
        return jsonify({"insight": insight_text.strip()})

    except Exception as e:
        return jsonify({"insight": f"Gemini engine was unable to compile insights at this time. Error: {str(e)}"}), 200

# --- Production Server Launcher ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
