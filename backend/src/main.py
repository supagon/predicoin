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

#SMA
    # @app.get("/btc_sma")
    # async def btcsmainfo():
    #     btcsmainf = get_btc_sma()
    #     return Response(content=btcsmainf, media_type='application/json')

    # @app.get("/eth_sma")
    # async def ethsmainfo():
    #     ethsmainf = get_eth_sma()
    #     return Response(content=ethsmainf, media_type='application/json')

@app.get("/btc_sma_plot")
async def btcsmaplot():
    return StreamingResponse(send_file('sma_output.jpg'),media_type='image/jpg')

@app.post("/btc_sma_post")
async def btcsmaprofit(investment_value : int):
    btcsmapro = post_btc_sma(investment_value)
    return {'investment':btcsmapro}

@app.post("/eth_sma_post")
async def ethsmaprofit(investment_value : int):
    ethsmapro = post_eth_sma(investment_value)
    return {'investment':ethsmapro}

@app.get("/eth_sma_plot")
async def ethsmaplot():
    return StreamingResponse(send_file('eth_sma_output.jpg'),media_type='image/jpg')

@app.post("/paxg_sma_post")
async def paxgsmaprofit(investment_value : int):
    paxgsmapro = post_paxg_sma(investment_value)
    return {'investment':paxgsmapro}

@app.get("/paxg_sma_plot")
async def paxgsmaplot():
    return StreamingResponse(send_file('paxg_sma_output.jpg'),media_type='image/jpg')

@app.post("/sol_sma_post")
async def solsmaprofit(investment_value : int):
    solsmapro = post_sol_sma(investment_value)
    return {'investment':solsmapro}

@app.get("/sol_sma_plot")
async def solsmaplot():
    return StreamingResponse(send_file('sol_sma_output.jpg'),media_type='image/jpg')

@app.post("/yfi_sma_post")
async def yfismaprofit(investment_value : int):
    yfismapro = post_yfi_sma(investment_value)
    return {'investment':yfismapro}

@app.get("/yfi_sma_plot")
async def yfismaplot():
    return StreamingResponse(send_file('yfi_sma_output.jpg'),media_type='image/jpg')

#MACD
# @app.get("/btc_macd")
# async def btcmacdinfo():
#     btcmacdinf = get_btc_macd()
#     return Response(content=btcmacdinf, media_type='application/json')

@app.post("/btc_macd")
async def btcmacdprofit(investment_value : int):
    btcmacdpro = post_btc_macd(investment_value)
    return {'investment':btcmacdpro}

@app.get("/btc_macd_plot")
async def btcmacdplot():
    return StreamingResponse(send_file('btc_macd_output.jpg'),media_type='image/jpg')

@app.post("/eth_macd")
async def ethmacdprofit(investment_value : int):
    ethmacdpro = post_eth_macd(investment_value)
    return {'investment':ethmacdpro}

@app.get("/eth_macd_plot")
async def ethmacdplot():
    return StreamingResponse(send_file('eth_macd_output.jpg'),media_type='image/jpg')

@app.post("/paxg_macd")
async def paxgmacdprofit(investment_value : int):
    paxgmacdpro = post_paxg_macd(investment_value)
    return {'investment':paxgmacdpro}

@app.get("/paxg_macd_plot")
async def paxgmacdplot():
    return StreamingResponse(send_file('paxg_macd_output.jpg'),media_type='image/jpg')

@app.post("/sol_macd")
async def solmacdprofit(investment_value : int):
    solmacdpro = post_sol_macd(investment_value)
    return {'investment':solmacdpro}

@app.get("/sol_macd_plot")
async def solmacdplot():
    return StreamingResponse(send_file('sol_macd_output.jpg'),media_type='image/jpg')

@app.post("/yfi_macd")
async def yfimacdprofit(investment_value : int):
    yfimacdpro = post_yfi_macd(investment_value)
    return {'investment':yfimacdpro}

@app.get("/yfi_macd_plot")
async def yfimacdplot():
    return StreamingResponse(send_file('yfi_macd_output.jpg'),media_type='image/jpg')


#RSI
@app.post("/btc_rsi_post")
async def btcrsiprofit(investment_value : int):
    btcrsipro = post_btc_rsi(investment_value)
    return {'investment':btcrsipro}

@app.get("/btc_rsi_plot")
async def btcrsiplot():
    return StreamingResponse(send_file('btc_rsi_output.jpg'),media_type='image/jpg')

@app.post("/eth_rsi_post")
async def ethrsiprofit(investment_value : int):
    ethrsipro = post_eth_rsi(investment_value)
    return {'investment':ethrsipro}

@app.get("/eth_rsi_plot")
async def ethrsiplot():
    return StreamingResponse(send_file('eth_rsi_output.jpg'),media_type='image/jpg')

@app.post("/paxg_rsi_post")
async def paxgrsiprofit(investment_value : int):
    paxgrsipro = post_paxg_rsi(investment_value)
    return {'investment':paxgrsipro}

@app.get("/paxg_rsi_plot")
async def paxgrsiplot():
    return StreamingResponse(send_file('paxg_rsi_output.jpg'),media_type='image/jpg')

@app.post("/sol_rsi_post")
async def solrsiprofit(investment_value : int):
    solrsipro = post_sol_rsi(investment_value)
    return {'investment':solrsipro}

@app.get("/sol_rsi_plot")
async def solrsiplot():
    return StreamingResponse(send_file('sol_rsi_output.jpg'),media_type='image/jpg')

@app.post("/yfi_rsi_post")
async def yfirsiprofit(investment_value : int):
    yfirsipro = post_yfi_rsi(investment_value)
    return {'investment':yfirsipro}

@app.get("/yfi_rsi_plot")
async def yfirsiplot():
    return StreamingResponse(send_file('yfi_rsi_output.jpg'),media_type='image/jpg')


## Bolinger Band
@app.post("/btc_bb_post")
async def btcbbprofit(investment_value : int):
    btcbbpro = post_btc_bb(investment_value)
    return {'investment':btcbbpro}

@app.get("/btc_bb_plot")
async def btcbbplot():
    return StreamingResponse(send_file('btc_bb_output.jpg'),media_type='image/jpg')

@app.post("/eth_bb_post")
async def ethbbprofit(investment_value : int):
    ethbbpro = post_eth_bb(investment_value)
    return {'investment':ethbbpro}

@app.get("/eth_bb_plot")
async def ethbbplot():
    return StreamingResponse(send_file('eth_bb_output.jpg'),media_type='image/jpg')

@app.post("/paxg_bb_post")
async def paxgbbprofit(investment_value : int):
    paxgbbpro = post_paxg_bb(investment_value)
    return {'investment':paxgbbpro}

@app.get("/paxg_bb_plot")
async def paxgbbplot():
    return StreamingResponse(send_file('paxg_bb_output.jpg'),media_type='image/jpg')

@app.post("/sol_bb_post")
async def solbbprofit(investment_value : int):
    solbbpro = post_sol_bb(investment_value)
    return {'investment':solbbpro}

@app.get("/sol_bb_plot")
async def solbbplot():
    return StreamingResponse(send_file('sol_bb_output.jpg'),media_type='image/jpg')

@app.post("/yfi_bb_post")
async def yfibbprofit(investment_value : int):
    yfibbpro = post_yfi_bb(investment_value)
    return {'investment':yfibbpro}

@app.get("/yfi_bb_plot")
async def yfibbplot():
    return StreamingResponse(send_file('yfi_bb_output.jpg'),media_type='image/jpg')


# @app.get("/test")
# def test():
#     test1 = original_vs_predict()
#     return Response(content=test1, media_type='application/json')


##############
### Manual ###
##############
@app.get("/save_result_sma_graph")
def save():
    plt_btc_sma()
    return {"status": "Success"}

@app.get("/save_model")
def save():
    train_model()
    return {"status": "Success"}

@app.get("/save_model_eth")
def save():
    train_model_eth()
    return {"status": "Success"}

@app.get("/save_model_paxg")
def save():
    train_model_paxg()
    return {"status": "Success"}

@app.get("/save_model_sol")
def save():
    train_model_sol()
    return {"status": "Success"}

@app.get("/save_model_yfi")
def save():
    train_model_yfi()
    return {"status": "Success"}

@app.get("/plot_btc_macd")
def plot():
    plot_macd()
    return {"status": "Success"}

@app.get("/plot_btc_rsi")
def plot():
    plot_btc_rsi()
    return {"status": "Success"}

@app.get("/plot_btc_bb")
def plot():
    btc_bb_plot()
    return {"status": "Success"}

@app.get("/plot_eth_sma")
def plot():
    plt_eth_sma()
    return {"status": "Success"}

@app.get("/plot_eth_macd")
def plot():
    plot_eth_macd()
    return {"status": "Success"}

@app.get("/plot_eth_rsi")
def plot():
    plot_eth_rsi()
    return {"status": "Success"}

@app.get("/plot_eth_bb")
def plot():
    plot_eth_bb()
    return {"status": "Success"}

@app.get("/plot_paxg_sma")
def plot():
    plt_paxg_sma()
    return {"status": "Success"}

@app.get("/plot_paxg_macd")
def plot():
    plot_paxg_macd()
    return {"status": "Success"}

@app.get("/plot_paxg_rsi")
def plot():
    plot_paxg_rsi()
    return {"status": "Success"}

@app.get("/plot_paxg_bb")
def plot():
    plot_paxg_bb()
    return {"status": "Success"}

@app.get("/plot_sol_sma")
def plot():
    plt_sol_sma()
    return {"status": "Success"}

@app.get("/plot_sol_macd")
def plot():
    plot_sol_macd()
    return {"status": "Success"}

@app.get("/plot_sol_rsi")
def plot():
    plot_sol_rsi()
    return {"status": "Success"}

@app.get("/plot_sol_bb")
def plot():
    plot_sol_bb()
    return {"status": "Success"}

@app.get("/plot_yfi_sma")
def plot():
    plt_yfi_sma()
    return {"status": "Success"}

@app.get("/plot_yfi_macd")
def plot():
    plot_yfi_macd()
    return {"status": "Success"}

@app.get("/plot_yfi_rsi")
def plot():
    plot_yfi_rsi()
    return {"status": "Success"}

@app.get("/plot_yfi_bb")
def plot():
    plot_yfi_bb()
    return {"status": "Success"}