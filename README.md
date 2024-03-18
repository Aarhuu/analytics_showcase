## Portfolio optimizer

Tool for S&P500 stock portfolio analysis. Based on mean-variance portfolio presented by Markovitz. Online hosted version available via [link](https://analytics-showcase.onrender.com/) 

Usage:

1. Select at least two stocks from the dropdown menu
2. Select time span of at least 15 days for historical returns and volatility data
3. Optimize Sharpe ratio, i.e., the optimal amount of expected return per volatility 
4. Select Value-at-risk (VAR) percentage and look forward period and simulate 

Tool based on python Dash web application. For local installation:

```
git clone https://github.com/Aarhuu/portfolio_optimizer.git
```

Use of virtual environment is recommended, i.e.,

```
python -m venv .venv
```

After virtual environment is created and activated, install requirements and run the app:

```
pip install -r requirements.txt

python app.py
```