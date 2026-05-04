#!/bin/bash
# Exemplos de uso da API Breast Cancer Classifier
# Pré-requisito: API rodando em http://127.0.0.1:8000
#   uvicorn api.main:app --reload

BASE_URL="http://127.0.0.1:8000"

echo "================================================"
echo "Health check"
echo "================================================"
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "================================================"
echo "Caso 1: Maligno (linha 1 do CSV - diagnostico real: M)"
echo "================================================"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 17.99,
    "texture_mean": 10.38,
    "perimeter_mean": 122.8,
    "area_mean": 1001.0,
    "smoothness_mean": 0.1184,
    "compactness_mean": 0.2776,
    "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471,
    "symmetry_mean": 0.2419,
    "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095,
    "texture_se": 0.9053,
    "perimeter_se": 8.589,
    "area_se": 153.4,
    "smoothness_se": 0.006399,
    "compactness_se": 0.04904,
    "concavity_se": 0.05373,
    "concave_points_se": 0.01587,
    "symmetry_se": 0.03003,
    "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38,
    "texture_worst": 17.33,
    "perimeter_worst": 184.6,
    "area_worst": 2019.0,
    "smoothness_worst": 0.1622,
    "compactness_worst": 0.6656,
    "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654,
    "symmetry_worst": 0.4601,
    "fractal_dimension_worst": 0.1189
  }' | python3 -m json.tool

echo ""
echo "================================================"
echo "Caso 2: Benigno (primeiro B do CSV - diagnostico real: B)"
echo "================================================"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 13.54,
    "texture_mean": 14.36,
    "perimeter_mean": 87.46,
    "area_mean": 566.3,
    "smoothness_mean": 0.09779,
    "compactness_mean": 0.08129,
    "concavity_mean": 0.06664,
    "concave_points_mean": 0.04781,
    "symmetry_mean": 0.1885,
    "fractal_dimension_mean": 0.05766,
    "radius_se": 0.2699,
    "texture_se": 0.7886,
    "perimeter_se": 2.058,
    "area_se": 23.56,
    "smoothness_se": 0.008462,
    "compactness_se": 0.01460,
    "concavity_se": 0.02387,
    "concave_points_se": 0.01315,
    "symmetry_se": 0.01980,
    "fractal_dimension_se": 0.00230,
    "radius_worst": 15.11,
    "texture_worst": 19.26,
    "perimeter_worst": 99.70,
    "area_worst": 711.2,
    "smoothness_worst": 0.1440,
    "compactness_worst": 0.1773,
    "concavity_worst": 0.2390,
    "concave_points_worst": 0.1288,
    "symmetry_worst": 0.2977,
    "fractal_dimension_worst": 0.07259
  }' | python3 -m json.tool

echo ""
echo "================================================"
echo "Caso 3: Maligno severo (linha 2 do CSV - diagnostico real: M)"
echo "================================================"
curl -s -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "radius_mean": 20.57,
    "texture_mean": 17.77,
    "perimeter_mean": 132.9,
    "area_mean": 1326.0,
    "smoothness_mean": 0.08474,
    "compactness_mean": 0.07864,
    "concavity_mean": 0.08690,
    "concave_points_mean": 0.07017,
    "symmetry_mean": 0.1812,
    "fractal_dimension_mean": 0.05667,
    "radius_se": 0.5435,
    "texture_se": 0.7339,
    "perimeter_se": 3.398,
    "area_se": 74.08,
    "smoothness_se": 0.005225,
    "compactness_se": 0.01308,
    "concavity_se": 0.01860,
    "concave_points_se": 0.01340,
    "symmetry_se": 0.01389,
    "fractal_dimension_se": 0.003532,
    "radius_worst": 24.99,
    "texture_worst": 23.41,
    "perimeter_worst": 158.8,
    "area_worst": 1956.0,
    "smoothness_worst": 0.1238,
    "compactness_worst": 0.1866,
    "concavity_worst": 0.2416,
    "concave_points_worst": 0.1860,
    "symmetry_worst": 0.2750,
    "fractal_dimension_worst": 0.08902
  }' | python3 -m json.tool
