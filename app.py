import os
import joblib
import math
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model missing at {MODEL_PATH}")

# Molecular Weights of Bases (g/mol)
BASE_WEIGHTS = {'A': 313.2, 'T': 304.2, 'C': 289.2, 'G': 329.2}

def get_molecular_weight(seq):
    return sum(BASE_WEIGHTS.get(base, 0) for base in seq)

def calculate_cpg_count(seq):
    # CpG dinucleotides are hotspots for mutation via deamination
    return seq.count("CG")

def check_palindrome(seq):
    if not seq or len(seq) < 2:
        return "No"
    # Complementary matching
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    rev_comp = "".join(comp.get(base, base) for base in reversed(seq))
    return "Yes (Self-Complementary Hairpin Risk)" if seq == rev_comp else "No"

def get_biochemical_profile(seq):
    if not seq:
        return {"purines": 0, "pyrimidines": 0, "keto": 0, "amino": 0}
    length = len(seq)
    # Purines (A, G) vs Pyrimidines (C, T)
    purines = seq.count('A') + seq.count('G')
    # Keto (G, T) vs Amino (A, C) hydrogen bonding profiles
    keto = seq.count('G') + seq.count('T')
    
    return {
        "purine_pct": round((purines / length) * 100, 1),
        "pyrimidine_pct": round(((length - purines) / length) * 100, 1),
        "hydrogen_bonding_profile": f"Keto: {round((keto/length)*100,1)}% | Amino: {round(((length-keto)/length)*100,1)}%"
    }

def analyze_advanced_mechanism(ref, alt):
    ref_len, alt_len = len(ref), len(alt)
    len_diff = abs(ref_len - alt_len)
    
    if ref_len == 1 and alt_len == 1:
        transitions = [{"A", "G"}, {"C", "T"}]
        is_trans = {ref, alt} in transitions
        mech = "Transition (Same Chemical Class)" if is_trans else "Transversion (Cross-Class Substitution)"
        consequence = "Point Mutation / Possible Missense or Synonymous"
        return mech, consequence
    
    # Indel checks
    frameshift = "Disruptive Frameshift (Alters downstream translation)" if len_diff % 3 != 0 else "In-frame Indel (Preserves codon reading frame)"
    if ref_len > alt_len:
        return f"Macro-Deletion (-{len_diff} Nucleotides)", frameshift
    else:
        return f"Macro-Insertion (+{len_diff} Nucleotides)", frameshift

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        ref = data.get("ref", "").upper().strip()
        alt = data.get("alt", "").upper().strip()
        
        if not ref or not alt:
            return jsonify({"status": "error", "message": "Invalid inputs"}), 400
            
        ref_len, alt_len = len(ref), len(alt)
        var_type = 0 if (ref_len == 1 and alt_len == 1) else (1 if ref_len > alt_len else (2 if ref_len < alt_len else 3))
        
        # ML Execution
        prediction = int(model.predict([[ref_len, alt_len, var_type]])[0])
        confidence = float(model.predict_proba([[ref_len, alt_len, var_type]])[0][prediction]) * 100
        
        # Deep Genomic Extractions
        mech, consequence = analyze_advanced_mechanism(ref, alt)
        ref_profile = get_biochemical_profile(ref)
        alt_profile = get_biochemical_profile(alt)
        
        # DNA Mass Delta Calculation
        mw_delta = get_molecular_weight(alt) - get_molecular_weight(ref)
        
        return jsonify({
            "status": "success",
            "prediction": "Pathogenic" if prediction == 1 else "Benign",
            "confidence": f"{confidence:.2f}%",
            "class": ["Single Nucleotide Variant (SNV)", "Deletion", "Insertion", "Complex Block substitution"][var_type],
            "mechanism": mech,
            "consequence": consequence,
            "mass_delta": f"{mw_delta:+.1f} g/mol",
            "ref_data": {
                "gc": round(((ref.count('G') + ref.count('C')) / ref_len) * 100, 1),
                "cpg": calculate_cpg_count(ref),
                "hairpin": check_palindrome(ref),
                "biochem": ref_profile
            },
            "alt_data": {
                "gc": round(((alt.count('G') + alt.count('C')) / alt_len) * 100, 1),
                "cpg": calculate_cpg_count(alt),
                "hairpin": check_palindrome(alt),
                "biochem": alt_profile
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
