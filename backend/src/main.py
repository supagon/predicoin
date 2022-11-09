from coin import *
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.get("/predict_btc")
def predict():
    prediction = get_new_prediction()
    return Response(content=prediction, media_type='application/json')

@app.get("/predict_eth")
def predict():
    prediction = eth_get_new_prediction()
    return Response(content=prediction, media_type='application/json')

@app.get("/predict_paxg")
def predict():
    prediction = paxg_get_new_prediction()
    return Response(content=prediction, media_type='application/json')

@app.get("/predict_sol")
def predict():
    prediction = sol_get_new_prediction()
    return Response(content=prediction, media_type='application/json')

@app.get("/predict_yfi")
def predict():
    prediction = yfi_get_new_prediction()
    return Response(content=prediction, media_type='application/json')

@app.get("/btc_info")
async def btcinfo():
    btcinf = get_btc_info()

    return Response(content=btcinf, media_type='application/json')

@app.get("/eth_info")
async def ethinfo():
    ethinf = get_eth_info()
    return Response(content=ethinf, media_type='application/json')

@app.get("/sol_info")
async def solinfo():
    solinf = get_sol_info()
    return Response(content=solinf, media_type='application/json')

@app.get("/yfi_info")
async def yfiinfo():
    yfiinf = get_yfi_info()
    return Response(content=yfiinf, media_type='application/json')

@app.get("/paxg_info")
async def paxginfo():
    paxginf = get_paxg_info()
    return Response(content=paxginf, media_type='application/json')

@app.get("/btc_sma")
async def btcsmainfo():
    btcsmainf = get_btc_sma()
    return Response(content=btcsmainf, media_type='application/json')

@app.get("/btc_sma_plot")
async def btcsmaplot():
    return StreamingResponse(send_file('sma_output.jpg'),media_type='image/jpg')

@app.post("/btc_sma")
async def btcsmaprofit(investment_value : int):
    btcsmapro = post_btc_sma(investment_value)
    return Response(content=btcsmapro, media_type='application/json')

@app.get("/btc_macd")
async def btcmacdinfo():
    btcmacdinf = get_btc_macd()
    return Response(content=btcmacdinf, media_type='application/json')

# @app.get("/test")
# def test():
#     test1 = original_vs_predict()
#     return Response(content=test1, media_type='application/json')

@app.get("/save_result_sma_graph")
def save():
    plt_btc_sma()
    return {"status": "Success"}

@app.post("/btc_macd")
async def btcmacdprofit(investment_value : int):
    btcmacdpro = post_btc_macd(investment_value)
    return Response(content=btcmacdpro, media_type='application/json')

