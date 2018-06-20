#python3
import aiohttp
import asyncio
import json
import time


def chunked_http_client(num_chunks, s):
    start = time.time()
    # Use semaphore to limit number of requests
    semaphore = asyncio.Semaphore(num_chunks)
    @asyncio.coroutine
    # Return co-routine that will work asynchronously and respect locking of semaphore
    def http_get(url, payload, verbose):
        nonlocal semaphore
        with (yield from semaphore):
            start = time.time()
            headers = {'content-type': 'application/json'}
            response = yield from s.request('post', url, data=json.dumps(payload), headers=headers)
            #if verbose: print("Response status:", response.status)
            body = yield from response.json()
            #if verbose: print(body)
            yield from response.wait_for_close()
            stop = time.time()
            print('time:', int((stop-start)*1000), 'ms')
        return body
    return http_get


def run_load_test(url, payloads, _session, concurrent, verbose):
    http_client = chunked_http_client(num_chunks=concurrent, s=_session)
    
    # http_client returns futures, save all the futures to a list
    tasks = [http_client(url, payload, verbose) for payload in payloads]

    dfs_route = []
    # wait for futures to be ready then iterate over them
    for future in asyncio.as_completed(tasks):
        data = yield from future
        try:
            dfs_route.append(data)
        except Exception as err:
            print("Error {0}".format(err))
    return dfs_route


if __name__=='__main__':
    NUM = 100
    CONCURRENT = 1
    VERBOSE = True
    payload = {10161: 1, 8706: 1, 7172: 1, 26120: 1, 32265: 1, 22026: 1, 22027: 1, 22034: 1, 23: 1, 22041: 1, 22042: 1, 12315: 1, 22048: 1, 22049: 1, 22050: 1, 7207: 1, 561: 1, 1078: 2, 22025: 1, 57: 1, 2229: 1, 24128: 1, 2116: 1, 9806: 1, 33208: 1, 10164: 1, 6238: 1, 610: 1, 12987: 1, 100: 1, 11297: 1, 12988: 1, 12989: 1, 15476: 1, 10942: 1, 5239: 1, 121: 2, 124: 1, 12992: 2, 12993: 1, 3209: 1, 3211: 1, 3864: 1, 150: 2, 12962: 1, 22043: 1, 678: 1, 12968: 1, 12969: 1, 7338: 1, 5292: 1, 12977: 1, 12978: 1, 12979: 1, 12980: 1, 1205: 2, 12982: 1, 12983: 1, 12984: 1, 12985: 1, 12986: 1, 8379: 1, 188: 1, 13002: 1, 702: 1, 12991: 1, 704: 1, 11296: 2, 12994: 1, 12995: 1, 12996: 1, 12997: 1, 12999: 2, 13000: 2, 13001: 1, 19146: 1, 13004: 1, 13006: 1, 13007: 1, 13008: 1, 13009: 1, 12972: 1, 9953: 1, 27874: 1, 27879: 2, 9965: 1, 9966: 1, 18159: 1, 12528: 1, 9971: 1, 24822: 1, 13575: 1, 13576: 1, 13577: 1, 13578: 1, 13580: 1, 13584: 1, 29457: 1, 13586: 1, 3863: 1, 13592: 1, 5404: 1, 26397: 1, 11551: 1, 19237: 1, 49: 1, 12585: 1, 12590: 1, 27951: 1, 29494: 1, 32649: 1, 12088: 1, 15165: 1, 12096: 1, 323: 1, 23606: 1, 326: 1, 12106: 1, 12107: 1, 12108: 1, 12109: 1, 12110: 1, 12112: 1, 12113: 1, 338: 2, 11700: 1, 11833: 1, 9057: 1, 9065: 1, 19309: 1, 33140: 1, 6517: 1, 9079: 1, 3448: 2, 19834: 2, 26491: 1, 19324: 1, 23423: 1, 15232: 2, 11145: 1, 11146: 1, 11147: 1, 396: 1, 29594: 1, 26524: 1, 26525: 1, 29599: 2, 29600: 1, 29601: 1, 29602: 2, 4276: 1, 29604: 1, 3498: 1, 29611: 1, 29612: 1, 10158: 1, 11697: 1, 11699: 1, 28084: 1, 11702: 1, 6071: 1, 11704: 1, 11707: 1, 11709: 1, 11710: 1, 11712: 1, 5059: 1, 29603: 1, 976: 1, 467: 1, 15319: 1, 12990: 1, 18912: 1, 336: 1, 3060: 1, 21493: 1, 10744: 1, 10745: 1, 510: 1}
    payload_list = [payload]*NUM
    end_point_recommend = "http://127.0.0.1:5000/recommend_old"
    with aiohttp.ClientSession() as session:  # We create a persistent connection
        start = time.time()
        loop = asyncio.get_event_loop()
        calc_routes = loop.run_until_complete(run_load_test(end_point_recommend, payload_list, session, CONCURRENT, VERBOSE))
        stop = time.time()
        print('Final time:', int((stop-start)*1000), 'ms')
