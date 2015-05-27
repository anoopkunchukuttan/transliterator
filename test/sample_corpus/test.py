
import yaml,pprint

config_params=yaml.load(open('supervised.yaml','r').read())

#print config_params['prior_config'].iteritems().next()
method,params=config_params['prior_config'].iteritems().next()

print method
pprint.pprint(params)
