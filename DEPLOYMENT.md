# Deployment Guide - Stock Market Intelligence Platform

## 🚀 Streamlit Cloud Deployment

### Prerequisites
1. GitHub account
2. Streamlit Cloud account (free at streamlit.io/cloud)
3. Git installed locally

### Step 1: Prepare Repository

```bash
cd projects/stock_market_intelligence

# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: Stock Market Intelligence Platform"
```

### Step 2: Create GitHub Repository

1. Go to GitHub.com
2. Click "New Repository"
3. Name: `stock-market-intelligence`
4. Description: "Enterprise-grade quantitative trading platform with LSTM forecasting"
5. **Do NOT** initialize with README (we already have one)
6. Click "Create Repository"

### Step 3: Push to GitHub

```bash
# Add remote
git remote add origin https://github.com/YOUR_USERNAME/stock-market-intelligence.git

# Push
git branch -M main
git push -u origin main
```

### Step 4: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Select your repository: `stock-market-intelligence`
4. Main file path: `deployment/app.py`
5. Python version: 3.8
6. Click "Deploy!"

**Deployment will take 3-5 minutes**

### Step 5: Configure Data Path (Important!)

Since the dataset is large, you have two options:

**Option A: Use Sample Data**
- Create a small sample dataset in the repository
- Update `src/data_loader.py` to use the sample

**Option B: Use External Data Source**
- Use `yfinance` to fetch data dynamically
- Update `data_loader.py` to fetch from Yahoo Finance API

---

## 📦 Local Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/stock-market-intelligence.git
cd stock-market-intelligence

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run deployment/app.py
```

### Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "deployment/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t stock-market-intelligence .
docker run -p 8501:8501 stock-market-intelligence
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file (not committed to git):

```env
DATA_PATH=/path/to/your/data
MODEL_PATH=./models
API_KEY=your_api_key_if_needed
```

### Streamlit Secrets

For Streamlit Cloud, add secrets in the dashboard:
1. Go to app settings
2. Click "Secrets"
3. Add your configuration

---

## ⚠️ Important Notes

### Data Files
- Dataset files are **NOT** included in git (too large)
- Models are **NOT** included (use Git LFS or separate storage)
- Only model metrics JSON files are tracked

### For Production Deployment
1. **Use Git LFS** for large model files
2. **Or** train models on deployment server
3. **Or** use cloud storage (S3, Google Cloud Storage)

### Model Training on Deployment

If deploying without pre-trained models:

```bash
# After deployment, run training script
python src/train_models.py
```

This will create all models in the `models/` directory.

---

## 📊 Performance Optimization

### For Streamlit Cloud (Free Tier)

1. **Reduce model size**: Use quantization
2. **Lazy loading**: Load models only when needed
3. **Caching**: Use `@st.cache_resource` for models
4. **Limit data**: Use subset of stocks for demo

### Example Optimization

```python
@st.cache_resource
def load_model(ticker):
    # Only load when needed
    model_path = f"models/{ticker.lower()}_lstm_model.h5"
    if os.path.exists(model_path):
        return load_model(model_path)
    return None
```

---

## 🐛 Troubleshooting

### Issue: "Module not found"
**Solution**: Ensure all dependencies in `requirements.txt`

### Issue: "Data files not found"
**Solution**: Update data paths in `src/data_loader.py`

### Issue: "Out of memory"
**Solution**: Reduce batch size, use smaller models

### Issue: "Slow loading"
**Solution**: Implement lazy loading, reduce model count

---

## 📝 Checklist Before Deployment

- [ ] README.md is complete and accurate
- [ ] requirements.txt has all dependencies
- [ ] .gitignore excludes large files
- [ ] Data paths are configurable
- [ ] Models are accessible (or training script included)
- [ ] Secrets/API keys not in code
- [ ] App runs locally without errors
- [ ] All tabs work correctly
- [ ] Performance is acceptable

---

## 🎯 Post-Deployment

1. **Test the deployed app** thoroughly
2. **Monitor performance** (Streamlit Cloud analytics)
3. **Update README** with live demo link
4. **Share on LinkedIn/Portfolio**
5. **Gather feedback** and iterate

---

## 📞 Support

If you encounter issues:
1. Check Streamlit Cloud logs
2. Review GitHub Actions (if using CI/CD)
3. Consult Streamlit documentation
4. Open an issue on GitHub

---

**Good luck with your deployment! 🚀**
