import sys, os
import argparse
import time
import random
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, Counter
from prometheus_client.exposition import basic_auth_handler

def auth_handler(url, method, timeout, headers, data):
    p_user=os.getenv('PUSH_USER') 
    p_password=os.environ.get('PUSH_PASSWORD')
    return basic_auth_handler(url, method, timeout, headers, data, p_user, p_password)

def make_registry():
    registry = CollectorRegistry()
    return registry

def make_gauge(g_name, g_description, g_registry):
    g = Gauge(g_name, g_description, registry=g_registry)
    return g

def set_gauge(gauge, g_value, job_label, g_registry, p_host='localhost:9091'):
    gauge.set(g_value)   # Set to a given value
    push_to_gateway(p_host, job=job_label, registry=g_registry, handler=auth_handler)

def main(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-t", "--push_host", action="store",
        required=True, dest="push_host", help="Host to push metrics to") 

    parser.add_argument("-u", "--push_user", action="store",
        required=True, dest="push_user", help="Push metric user") 

    parser.add_argument("-p", "--push_password", action="store",
        required=True, dest="push_password", help="Push metric password")   

    args = parser.parse_args()

    os.environ['PUSH_USER'] = args.push_user
    os.environ['PUSH_PASSWORD'] = args.push_password

    print('using metrics host', args.push_host)
    
    g_registry = make_registry()
    random_gauge = make_gauge('job_random_testval', "A random value to test", g_registry)

    while True:
        g_value = random.random()
        set_gauge(random_gauge, g_value, "random_job_label", g_registry, p_host=args.push_host)
        time.sleep(2)


if __name__ == "__main__":
    main(sys.argv[1:]) 
    