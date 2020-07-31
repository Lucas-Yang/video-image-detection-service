import hashlib
import http.client as httplib
import json
import os
import shlex
import subprocess
import time
import urllib
from io import BytesIO
from typing import Any
from urllib.parse import urlparse, quote

import requests
from PIL import Image

from ..keywords.keywordgroup import KeywordGroup


class RequestNoContentError(Exception):
    '''Exception'''


# new
class _Tools(KeywordGroup):

    # Public
    def get_url_resp(self, url, method, **kwargs):
        """  Get response of an api
        | *Option*            | *Man.* | *Description*           |
        | url                 | Yes    | api url, host and path  |
        | method              | Yes    | request method          |
        | other params        | No     | request params          |

        Examples:
        |    Get Url Resp    |    http://127.0.0.1/wd/hub/status   |    GET    |
        """
        param = {}
        param.update(kwargs)
        resp = ""
        if method.lower() == "get":
            resp = self._get(url, params=param, timeout=10)
        elif method.lower() == "post":
            for (k, v) in param.items():
                if str(v).lower() == "true":
                    param[k] = True
                elif str(v).lower() == "false":
                    param[k] = False
            param = json.dumps(param)
            self._info("req: " + param)
            resp = self._post(url, data=param, timeout=10)
        else:
            raise AssertionError("Request Error, please check your data!")
        return resp.text

    def device_proxy_status(self, device_id: str) -> None:
        _cmd = 'adb -s {} shell dumpsys settings | grep global_http_proxy'.format(device_id)
        subprocess.call(shlex.split(_cmd))

    def set_device_proxy(self, device_id: str, host: str = '', port: int = 0) -> None:
        _cmd = 'adb -s {0} shell settings put global http_proxy {1}:{2}'.format(device_id, host, port)
        subprocess.call(shlex.split(_cmd))
        self.device_proxy_status(device_id)

    def remove_device_proxy(self, device_id: str) -> None:
        self.set_device_proxy(device_id)

    def mock_api_rule(self, **kwargs: Any) -> str:
        _hassan = 'http://hassan.bilibili.co/ep/admin/hassan/v2/mock/rule/add-auto'
        _method = 'POST'
        expected_keys = {'env_id', 'host', 'path', 'state', 'fuzz_mode', 'response', 'white_list', 'black_list',
                         'is_grpc', 'mock_opportunity'}
        required_keys = {'env_id', 'host', 'path'}
        # intersection
        if (set(kwargs.keys()) & expected_keys) < expected_keys:
            print('WARN: Some keys are missing, using default values.')
        if (set(kwargs.keys()) & required_keys) < required_keys:
            raise RuntimeError("Key 'env_id', 'host' and 'path' must be valid.")
        if kwargs.get('white_list') and kwargs.get('black_list') and \
                (set(kwargs.get('white_list')) & set(kwargs.get('black_list'))):
            raise RuntimeError('Duplicate keys in white/black lists.')
        _args = {
            'env_id': str(kwargs.get('env_id')).strip(),
            'path': str(kwargs.get('path')).strip(),
            'state': 'true' == str(kwargs.get('state')).lower(),
            'fuzz_mode': 3 if kwargs.get('fuzz_mode') is not None and 3 == int(kwargs.get('fuzz_mode')) else 0,
            'response': str(kwargs.get('response')) if kwargs.get('response') is not None else '{}',
            'white_list': list(kwargs.get('white_list')) if kwargs.get('white_list') is not None else [],
            'black_list': list(kwargs.get('black_list')) if kwargs.get('black_list') is not None else [],
            'is_grpc': kwargs.get('is_grpc') if kwargs.get('is_grpc') is not None else 0,
            'mock_opportunity': 2 if kwargs.get('mock_opportunity') is not None and 2 == int(
                kwargs.get('mock_opportunity')) else 1,
        }
        tmp_host = str(kwargs.get('host'))
        if tmp_host == 'app.bilibili.com':
            # handle downgrade
            tmp_host = 'app.bilibili.com|app.biliapi.com|app.biliapi.net'
        _args.update({'host': tmp_host})
        _args = json.dumps(_args)
        self._info(_args)
        _res = requests.post(url=_hassan, data=_args, timeout=20).text
        print(_res)
        return _res

    def get_api_param(self, param, url, method, **kwargs):
        ''' Get API response of a param
        :param param: String
        :param url: String
        :param method: String
        :param kwargs: Default Null
        :return: String
        '''
        start_time = time.time()
        r = {}
        while (time.time() - start_time <= 10):
            content = self.get_url_resp(url, method, **kwargs)
            r = json.loads(content)
            if r.get('data', {}).get(param, '-1') == '-1':
                time.sleep(2)
            else:
                break
        return r.get('data', {}).get(param, '-1')

    def include_str(self, sub, all):
        if sub in all:
            return True
        else:
            return False

    # Private
    def _get(self, *args, **kwargs):
        return requests.get(*args, **kwargs)

    def _post(self, *args, **kwargs):
        return requests.post(*args, **kwargs)

    def make_sign_android(self,mid, value: dict):

        url = "http://hassan.bilibili.co/ep/admin/hassan/v2/uat/account/cookie/query"
        payload = json.dumps({'mid':int(mid)})
        headers = {
            'content-type': "application/json",
            'cache-control': "no-cache",
        }

        res = requests.request("POST", url, data=payload, headers=headers)
        print(res.json())
        access_key = res.json()['data']['token']
        print(access_key)
        # access_key = 'c2ed53a74eeefe3cf99fbd01d8c9c375'
        value['access_key'] = access_key
        value['ts'] = int(time.time())
        value['device'] = 'phone'
        value['mobi_app'] = 'android'
        value['platform'] = 'android'

        value['actionKey'] = 'appkey'
        if value.get('appkey') is None:
            value['appkey'] = '5245acedbb88a3e2'
        if value.get('appsecret') is None:
            value['appsecret'] = '66998d7f276f47b0fd31dc026c9f22cf'

        val_keys = [key for key in value.keys()]
        val_keys.sort()
        val = {}
        for key in val_keys:
            val[key] = value[key]
        val_str = urllib.parse.urlencode(val, quote_via=quote)
        md5_str = val_str + access_key

        md5 = hashlib.md5()
        md5.update(md5_str.encode('utf8'))
        value['sign'] = md5.hexdigest()
        val_keys.append('sign')
        val_keys.sort()
        da = {}
        for key in val_keys:
            da[key] = value[key]
        return da

    def req_app_bilibili(self,url,method,mid,data):
        params = self.make_sign_android(mid,data)
        proxies = {'http':'http://172.22.33.245','https':'https://172.22.33.245'}
        if method == 'get':
            r = requests.get(url, params=params,proxies=proxies,verify=False)
        else:
            r = requests.post(url, data=params,proxies=proxies,verify=False)

        print(r.url)
        res = r.json()
        print(res)
        return res

    def request_account_mine(self, build):
        url = "http://app.bilibili.com/x/v2/account/mine"
        res = self.req_app_bilibili(url, {"build": build})

        sections = res['data']['sections']
        return_list = []
        for i in sections:
            # print(i["title"]+":")
            for n in i['items']:
                # print(n['title'])
                return_list.append(n['title'])
        return return_list

    def upload_file(self, dict_path):
        file_list = os.listdir(dict_path)
        bfs_urls_list = []
        if file_list:
            for each_png_name in file_list:
                if each_png_name.endswith('png'):
                    each_data = {}
                    each_png_path = os.path.join(dict_path, each_png_name)
                    each_bfs_url = _upload_image(each_png_path)
                    each_data.update({"casename": each_png_name.strip(".png"), "bfsurl": each_bfs_url})
                    bfs_urls_list.append(each_data)
        print("Total Files: ", len(bfs_urls_list))
        print(bfs_urls_list)
        len_failed_bfs_list = [x for x in bfs_urls_list if x['bfsurl'] == "upload failed"]
        print("Upload failed: ", len(len_failed_bfs_list))
        if not len_failed_bfs_list:
            print("All files Upload successed!!")
            return bfs_urls_list
        else:
            print(len_failed_bfs_list)
            raise Exception("有文件上传失败重新上传")

    def post_pic_diff(self, version, module, bfs_urls, app, devicename):
        url = "http://rome.bilibili.co/v/picdiff"
        sessionid = str(int(time.time()))
        common = {"app": app, "device": devicename, "version": version}
        data = json.dumps({"sessionid": sessionid, "module": module, "common": common, "data": bfs_urls})
        r = requests.post(url, data)
        print(r.content)


# 上传图片
def _upload_image(fpath):
    image_io = resize_image(fpath)
    bfs_url = _bfs_upload_by_io(image_io)
    # bfs_url = _bfs_upload_by_type(fpath, "image/png")
    for i in range(3):
        if not bfs_url:
            bfs_url = _bfs_upload_by_io(image_io)
            print(fpath + "try: " + str(i))
            time.sleep(0.3)
        else:
            break
    bfs_url = (bfs_url if bfs_url else "upload failed")
    return bfs_url


def resize_image(fpath):
    im = Image.open(fpath)
    # im.show()
    w, h = im.size[0], im.size[1]

    new_img = im.resize((int(w * 0.8), int(h * 0.8)), Image.ANTIALIAS)
    # new_img.save("testsize.png")
    imgByteArr = BytesIO()  # 创建一个空的Bytes对象

    new_img.save(imgByteArr, format='PNG')  # PNG就是图片格式，我试过换成JPG/jpg都不行

    imgByteArr = imgByteArr.getvalue()  # 这个就是保存的图片字节流
    return imgByteArr


def _bfs_upload_by_io(im):
    uri = "/bfs/davinci"
    res = ""
    try:
        body = im
        headers = {"Content-type": "image/png"}

        httpclient = httplib.HTTPConnection("uat-bfs.bilibili.co", 80, timeout=30)
        httpclient.request("PUT", uri, body, headers)
        response = httpclient.getresponse()
        res = (response.getheader('Location') if response.status == 200 else None)
        print('Merge_Location: ', res)
    except Exception:
        print('upload fail')
        if response:
            print('response', response.read().decode('utf8'))

    finally:
        if httpclient:
            httpclient.close()
    return res
