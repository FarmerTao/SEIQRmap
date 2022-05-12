import requests

def download(url):
    req = requests.get(url)
    filename = url.split('/')[-1]
    if req.status_code != 200:
        print('下载异常')
        return
    try:
        with open(filename, 'wb') as f:
            #req.content为获取html的内容
            f.write(req.content)
    except Exception as e:
        print(e)
        
url ='https://covid.ourworldindata.org/data/owid-covid-data.csv'
download(url)