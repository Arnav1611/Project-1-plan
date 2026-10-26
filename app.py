"""Run from this folder: python -m streamlit run app.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pipeline import load_config, manifest, out, predict_image

st.set_page_config(page_title='MRI CDR-group research demo', layout='wide')
st.title('MRI CDR-group classification')
st.caption('Academic research prototype. Not clinically validated.')
st.write('Select an authorised, processed sample to view its MRI slices and model prediction.')
config_path = st.sidebar.text_input('Configuration file', 'config.yaml')
try:
    c = load_config(config_path)
    selected = json.loads(out(c, 'models/selected.json').read_text())['model']
    df = manifest(c)
    st.sidebar.write('Model selected using development data:', selected)
    st.sidebar.caption('Samples are shown without their clinical labels. Prediction does not use CDR.')
    # Keep final-test samples hidden until evaluation is complete.
    if not out(c, 'results/test_metrics.csv').exists():
        df = df.query("split == 'development'")
    sid = st.selectbox('Sample', df.subject_id.tolist())
    if st.button('Analyse MRI', type='primary'):
        row = df.set_index('subject_id').loc[sid]
        with st.spinner('Loading MRI and running the selected model…'):
            result, slices = predict_image(c, row.image_path)
        group = 'CDR-positive group' if result['label'] else 'CDR-zero group'
        a, b, d = st.columns(3)
        a.metric('Research prediction', group)
        b.metric('CDR-positive model score', f"{result['score']:.3f}")
        d.metric('Time including model loading', f"{result['seconds_including_model_loading']:.2f} s")
        st.caption('The score is not a validated probability of disease. These are preprocessed slices.')
        st.image([np.ascontiguousarray(x.T[::-1]) for x in slices],
                 caption=[f'Slice {i+1}' for i in range(len(slices))], width=180, clamp=True)
    with st.expander('Development comparison'):
        st.dataframe(pd.read_csv(out(c, 'results/development_comparison.csv')), hide_index=True)
    if out(c, 'results/test_metrics.csv').exists():
        with st.expander('Locked held-out evaluation'):
            st.dataframe(pd.read_csv(out(c, 'results/test_metrics.csv')), hide_index=True)
except Exception as exc:
    st.error(str(exc))
    st.info('Complete the README steps in order. Dataset files and trained weights are not included.')
