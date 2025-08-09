# AI-Powered Analytics & Insights Dashboard (Streamlit)

Blend GA4 funnels/pathing + customer sentiment + GPT summaries into one brandable dashboard.

## Quick Deploy (Streamlit Cloud)
1. Create a public GitHub repo and add these files:
   - `streamlit_app.py`
   - `requirements.txt`
   - (optional) sample CSVs in `/samples`
2. In Streamlit Cloud: **Deploy → From GitHub**. Set **Main file path** to `streamlit_app.py`.
3. In **App Settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
4. Reload the app. Start in **Demo Data** mode, then upload your CSVs.

## CSV Schemas
- GA4 Page Metrics: `date, page, device, sessions, bounces, view_item, add_to_cart, purchase`
- Pathing: `date, from, to, count`
- Reviews: `date, rating, sentiment, theme, text`

## Local Dev
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

© 2025 Sense Data Lab
