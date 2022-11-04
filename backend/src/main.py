from coin import *
from fastapi import FastAPI, Response

app = FastAPI()

@app.get("/predict_btc")
def predict():
    prediction = get_new_prediction()
    return Response(content=prediction, media_type='application/json')

@app.get("/btc_info")
def btcinfo():
    btcinf = get_btc_info()
    return Response(content=btcinf, media_type='application/json')

@app.get("/eth_info")
def ethinfo():
    ethinf = get_eth_info()
    return Response(content=ethinf, media_type='application/json')

@app.get("/sol_info")
def solinfo():
    solinf = get_sol_info()
    return Response(content=solinf, media_type='application/json')

@app.get("/yfi_info")
def yfiinfo():
    yfiinf = get_yfi_info()
    return Response(content=yfiinf, media_type='application/json')

@app.get("/paxg_info")
def paxginfo():
    paxginf = get_paxg_info()
    return Response(content=paxginf, media_type='application/json')

@app.get("/btc_sma")
def btcsmainfo():
    btcsmainf = get_btc_sma()
    return Response(content=btcsmainf, media_type='application/json')

@app.post("/btc_sma")
def btcsmaprofit(investment_value : int):
    btcsmapro = post_btc_sma(investment_value)
    return Response(content=btcsmapro, media_type='application/json')

@app.get("/btc_macd")
def btcmacdinfo():
    btcmacdinf = get_btc_macd()
    return Response(content=btcmacdinf, media_type='application/json')

# @app.get("/test")
# def test():
#     test1 = original_vs_predict()
#     return Response(content=test1, media_type='application/json')

# @app.get("/save_result")
# def save():
#     save_result()
#     return {"status": "Success"}

@app.post("/btc_macd")
def btcmacdprofit(investment_value : int):
    btcmacdpro = post_btc_macd(investment_value)
    return Response(content=btcmacdpro, media_type='application/json')

