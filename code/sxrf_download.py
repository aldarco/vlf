from bs4 import BeautifulSoup
import requests
import urllib.request as urlreq
import time
import datetime as dt
from sunpy import timeseries as ts
from sunpy.net import Fido
from sunpy.net import attrs as a
import pandas as pd
from io import StringIO
import warnings

def download_xrayfluxdata(dates,  outputpath=None, delay=0):
    '''
    srapping Xray Flux data from CONIDA Space Weather Web
    Input:
    - dates: list or single datetime values 
    - outputpath: optional, if None, data is not saved, else data filesare saved in specified path

    Output:
    - time series of Xrayflux

    '''

    url = "https://climaespacial.conida.gob.pe/GOES/DatosClimaEspacial/AnalizarDatoXrays_Recolectar.php?index=6&fecha={}"
    print("URL: {}".format(url))
    if type(dates) != list: dates = [dates]
    df_xr = pd.DataFrame()
    for d in  dates:
        print("Downloading {}...".format(d.date()), end=" ")
        url1 = url.format(d.date())

        response = requests.get(url1)
        soup = BeautifulSoup(response.text, 'html.parser')

        csv = soup.findAll('pre')[0].find(text=True)
        dfcsv = pd.read_csv(StringIO(csv), sep=",",parse_dates=[0])
        df_xr = pd.concat([df_xr, dfcsv])
        if outputpath:
            outputname = "XRF_{}.csv".format(d.strftime("%d%m%Y"))
            df_xr.to_csv(outputpath+"/"+outputname)
            
            print("> Done.".format(d.date()), end="\n")
        time.sleep(delay)

    print("\nCompleted.")
    return df_xr

def sunpy_xrf(tstart, tend, sat_number=16, outputpath=None):
    '''
    Function to download data (Solar x-ray flux) using SunPy library
    Input arguments: 
    - tstart: str formatted start date, eg. "YYYY-MM-DD HH:mm"
    - tend: str formatted end date, eg. "YYYY-MM-DD HH:mm"
    - sat_number: int satellite number. Default 16. 
    Output:
        dataframe of time series
        if outputpath is given, CSV files of each day with the data
        
    '''
    #outputpath = "/data/PAVNET/2023/PLO_191023-021123/xrayflux/"
    
    result = Fido.search(a.Time(tstart, tend), a.Instrument("XRS"), a.Resolution("avg1m"),a.goes.SatelliteNumber(sat_number))
    #print(result)
    
    goes_files = Fido.fetch(result)
    goes_data = ts.TimeSeries(goes_files, concat=True)
    dfout = pd.DataFrame()
    for g in goes_data:
        date_ = "".join(str(g.time[0].to_datetime().date()).split("-")[::-1])
        df = g.to_dataframe()
        # names to params/cols
        c = list(df.columns)
        c[0], c[1] = g.observatory+'_short', g.observatory+'_long'
        df.columns = c
        df.index.name = "DateTime"
        if outputpath: # guardar por día ed dato
            fileout = f'{outputpath}/XRF_{date_}.csv'
            df.to_csv(fileout, columns=[c[0], c[1]])
            print(f"> {fileout} done")
        
        dfout = pd.concat([dfout, df])
    if len(dfout)==0: warnings.warn("Got no data. Try using another satellite number, e.g. 18.")    
        
    return dfout 


if __name__ == "__main__":
    pass
