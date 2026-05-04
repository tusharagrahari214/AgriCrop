#!/usr/bin/env python3
# ================================================================================
# AgriCrop — Inference Script
# Reads JSON from stdin, runs XGBoost prediction + advisory, prints JSON to stdout
# Called from Node.js via child_process
# ================================================================================

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.join(SCRIPT_DIR, "artifacts")

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

CROP_DISPLAY = {
    'rice': 'Rice', 'maize': 'Maize', 'chickpea': 'Chickpea',
    'kidneybeans': 'Kidney Beans', 'pigeonpeas': 'Pigeon Peas',
    'mothbeans': 'Moth Beans', 'mungbean': 'Mung Bean',
    'blackgram': 'Black Gram', 'lentil': 'Lentil',
    'pomegranate': 'Pomegranate', 'banana': 'Banana', 'mango': 'Mango',
}

CROP_EMOJI = {
    'Rice': '🍚', 'Maize': '🌽', 'Chickpea': '🫘', 'Kidney Beans': '🫘',
    'Pigeon Peas': '🫘', 'Moth Beans': '🫘', 'Mung Bean': '🫘',
    'Black Gram': '🫘', 'Lentil': '🫘', 'Pomegranate': '🍎',
    'Banana': '🍌', 'Mango': '🥭',
}

# Load saved model artifacts
model   = joblib.load(os.path.join(ARTIFACT_DIR, "xgb_model.pkl"))
scaler  = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))


def run_inference(input_dict):
    """Run crop prediction + advisory."""
    N           = float(input_dict['N'])
    P           = float(input_dict['P'])
    K           = float(input_dict['K'])
    temperature = float(input_dict['temperature'])
    humidity    = float(input_dict['humidity'])
    ph          = float(input_dict['ph'])
    rainfall    = float(input_dict['rainfall'])

    X_raw = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURE_COLS)
    enc   = model.predict(scaler.transform(X_raw))[0]
    crop  = CROP_DISPLAY.get(encoder.inverse_transform([int(enc)])[0], '?')
    emoji = CROP_EMOJI.get(crop, '🌱')

    # ── Nitrogen ──
    if   N < 20:  n_status, n_level, n_advice = "CRITICAL", "critical", "Severe N deficiency. Apply urea 60 kg/ha. Chlorosis risk. Immediate foliar spray."
    elif N < 40:  n_status, n_level, n_advice = "LOW", "low", "Apply ammonium nitrate or DAP. Monitor leaf colour. Top-dress 30 kg N/ha."
    elif N <= 80: n_status, n_level, n_advice = "OPTIMAL", "optimal", "N in ideal range. Maintain current regimen."
    elif N < 120: n_status, n_level, n_advice = "HIGH", "high", "Reduce N inputs. Excessive vegetative growth risk. No further N this season."
    else:         n_status, n_level, n_advice = "EXCESS", "excess", "Leaching risk. No N application. Reduce irrigation."

    # ── Phosphorus ──
    if   P < 15:  p_status, p_level, p_advice = "LOW", "low", "Apply SSP or DAP. Root development impaired. Apply 40 kg P2O5/ha."
    elif P <= 60: p_status, p_level, p_advice = "OPTIMAL", "optimal", "P adequate for root development and energy transfer."
    else:         p_status, p_level, p_advice = "HIGH", "high", "Excess P. Avoid P fertilisers. May lock out Zn and Fe."

    # ── Potassium ──
    if   K < 15:  k_status, k_level, k_advice = "LOW", "low", "Apply MOP 40 kg K2O/ha. Low K reduces water uptake and disease resistance."
    elif K <= 80: k_status, k_level, k_advice = "OPTIMAL", "optimal", "K in ideal range. Maintains plant vigour."
    else:         k_status, k_level, k_advice = "HIGH", "high", "Excess K. Avoid MOP. May cause Ca/Mg imbalance."

    # ── pH ──
    if   ph < 5.5:  ph_status, ph_level, ph_advice = "ACIDIC", "critical", "Apply ag lime 2-4 t/ha. Nutrient availability severely reduced. Target 6.0-6.5."
    elif ph < 6.0:  ph_status, ph_level, ph_advice = "SLIGHTLY ACIDIC", "low", "Light lime 1-2 t/ha. Target pH 6.0-7.0."
    elif ph <= 7.5: ph_status, ph_level, ph_advice = "OPTIMAL", "optimal", "Soil pH in ideal range. Nutrient availability maximised."
    elif ph <= 8.5: ph_status, ph_level, ph_advice = "ALKALINE", "high", "Apply elemental sulfur or ammonium sulfate. Reduces Fe, Mn, Zn, B availability."
    else:           ph_status, ph_level, ph_advice = "STRONGLY ALKALINE", "excess", "Severe imbalance. Consult agronomist. Apply gypsum + sulfur urgently."

    # ── Irrigation ──
    if   humidity < 40:
        irr_status, irr_level, irr_advice = "LOW HUMIDITY", "low", "Increase irrigation. Moisture stress risk. Consider drip irrigation."
    elif humidity <= 70:
        irr_status, irr_level, irr_advice = "OPTIMAL", "optimal", "Moisture levels support healthy crop growth."
    else:
        irr_status, irr_level, irr_advice = "HIGH HUMIDITY", "high", "Reduce irrigation. Fungal disease risk. Ensure drainage."

    if   rainfall < 50:  irr_advice += " Critically low rainfall — full supplemental irrigation required."
    elif rainfall < 100: irr_advice += " Below-adequate rainfall — partial irrigation support needed."
    elif rainfall > 250: irr_advice += " Heavy rainfall — ensure drainage to prevent waterlogging."

    # ── Health Status ──
    non_opt  = sum([n_status != "OPTIMAL", p_status != "OPTIMAL", k_status != "OPTIMAL", ph_status != "OPTIMAL"])
    critical = n_status == "CRITICAL" or ph_status in ("ACIDIC", "STRONGLY ALKALINE")
    if critical or non_opt >= 3:
        health, health_icon, health_level = "DEFICIENT", "⚠", "critical"
    elif non_opt >= 1:
        health, health_icon, health_level = "STRESSED", "⚡", "warning"
    else:
        health, health_icon, health_level = "HEALTHY", "✔", "healthy"

    return {
        'crop': crop,
        'cropEmoji': emoji,
        'health': health,
        'healthIcon': health_icon,
        'healthLevel': health_level,
        'input': {
            'N': N, 'P': P, 'K': K,
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'ph': round(ph, 2),
            'rainfall': round(rainfall, 1),
        },
        'advisories': {
            'nitrogen':   {'status': n_status,   'level': n_level,   'advice': n_advice},
            'phosphorus': {'status': p_status,   'level': p_level,   'advice': p_advice},
            'potassium':  {'status': k_status,   'level': k_level,   'advice': k_advice},
            'ph':         {'status': ph_status,  'level': ph_level,  'advice': ph_advice},
            'irrigation': {'status': irr_status, 'level': irr_level, 'advice': irr_advice},
        }
    }


if __name__ == '__main__':
    try:
        raw = sys.stdin.read()
        input_data = json.loads(raw)

        # Support single or batch predictions
        if isinstance(input_data, list):
            results = [run_inference(item) for item in input_data]
            print(json.dumps(results, ensure_ascii=False))
        else:
            result = run_inference(input_data)
            print(json.dumps(result, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({'error': str(e)}, ensure_ascii=False))
        sys.exit(1)
