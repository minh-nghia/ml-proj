import pprint
import numpy as np
import json
import math
import time

np.set_printoptions(threshold='nan')
    
with open('logs.json', 'r') as f:
    data = json.load(f)

with open('alerts.json', 'r') as f:
    alerts = json.load(f)

with open('zone_convert.json', 'r') as f:
    zones = json.load(f)

with open('country_convert.json', 'r') as f:
    countries = json.load(f)

alert_dict = {a['event']['id']: a['policy']['policy_name'] for a in alerts}

py_data = []

for d in data:
    if d['locations']['resource'] == 'UNKNOWN' or d['locations']['request'] == 'UNKNOWN':
        continue
    if d['operation'] == 'UNKNOWN' or d['principal'] == 'UNKNOWN' or d['resource'] == 'UNKNOWN':
        continue
    loc = {
        'request': {
            'ip': 'UNKNOWN',
            'zone': 'UNKNOWN',
            'country': 'UNKNOWN'
        },
        'resource': {
            'ip': 'UNKNOWN',
            'zone': 'UNKNOWN',
            'country': 'UNKNOWN'
        }
    }
    rqloc = d['locations']['request'].split('://')
    rqloc_keys = rqloc[0].split('+')
    rqloc_items = rqloc[1].split(';')
    for i in range(len(rqloc_keys)):
        loc['request'][rqloc_keys[i]] = rqloc_items[i]
        
    rsloc = d['locations']['resource'].split('://')
    rsloc_keys = rsloc[0].split('+')
    rsloc_items = rsloc[1].split(';')
    for i in range(len(rsloc_keys)):
        loc['resource'][rsloc_keys[i]] = rsloc_items[i]

    try:
        ltime = time.strptime(d['time'], '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        ltime = time.strptime(d['time'], '%Y-%m-%dT%H:%M:%S')
        
    log = {
        'operation': d['operation'],
        'principal': d['principal'],
        'resource': d['resource']['type'],
        'time': time.mktime(ltime),
        'request_ip': loc['request']['ip'],
        'request_zone': loc['request']['zone'],
        'request_country': loc['request']['country'],
        'resource_ip': loc['resource']['ip'],
        'resource_zone': loc['resource']['zone'],
        'alerted': d['id'] in alert_dict.keys(),
    }
    py_data.append(log)

resource_pool = []
operation_pool = []
principal_pool = []
resource_ip_pool = []
resource_zone_pool = []
resource_country_pool = []
request_ip_pool = []
request_zone_pool = []
request_country_pool = []


def get_index(pool, value, zeros=['UNKNOWN']):
    if value in zeros:
        return 0
    else:
        pool_keys = [p[0] for p in pool]
        if value not in pool_keys:
            pool.append([value, 1])
            pool_keys = [p[0] for p in pool]
        else:
            pool[pool_keys.index(value)][1] += 1
        return pool_keys.index(value) + 1

rows = []

for log in py_data:
    row = []
    row.append(get_index(principal_pool, log['principal']))
    row.append(get_index(operation_pool, log['operation']))
    row.append(get_index(resource_pool, log['resource']))
    row.append(get_index(request_ip_pool, log['request_ip'], ['UNKNOWN', 'UNSPECIFIED']))
    row.extend(countries[log['request_country']])
    row.extend(zones[log['resource_zone']])
    row.append(math.cos(2*math.pi*log['time']/86400))
    row.append(math.sin(2*math.pi*log['time']/86400))
    row.append(math.cos(2*math.pi*log['time']/86400/7))
    row.append(math.sin(2*math.pi*log['time']/86400/7))
    row.append(-1 if log['alerted'] else 1)
    rows.append(row)    

print(row)

print(principal_pool, len(principal_pool))
print(operation_pool, len(operation_pool))
print(resource_pool, len(resource_pool))
print(request_ip_pool, len(request_ip_pool))
array = np.array(rows, dtype='float32')

with open('data.npy', 'wb') as npf:
    np.save(npf, array)
