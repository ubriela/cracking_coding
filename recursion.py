"""
Flatten dictionary

{
  'Key1': '1',
  'Key2': {
    'a' : '2', // Key2.a -> '2'
    'b' : '3',
    'c' : {
      'd' : '3',
      'e' : '1'
      }
    }
}


{
  'Key1': '1',
  'Key2.a': '2',
  'Key2.b' : '3',
  'Key2.c.d' : '3',
  'Key2.c.e' : '1'
}

for all key,val:
   if val is a dict: # Key2
   else
      // return a set of keys,values if we cannot further open
      return Key2.c.d

"""

def flatten_dict(d, key):
    res = dict()
    for k, v in d.items():
        if type(v) is not dict:
            res[key + '.' + k] = v
        else:
            res_dict = flatten_dict(v, key + '.' + k)
            for k2,v2 in res_dict.items():
                res[k2] = v2

    return res

# input_d = dict(
# {
#   'Key1': '1',
#   'Key2': {
#     'a' : '2',
#     'b' : '3',
#     'c' : {
#       'd' : '3',
#       'e' : '1'
#       }
#     }
# })
# print(flatten_dict(input_d, ""))