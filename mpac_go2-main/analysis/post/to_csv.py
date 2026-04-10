import numpy as np
import h5py
import os
import sys


filename = 'data/latest.h5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

file = h5py.File(filename,'r')
data = file['time_series']
attr = file['attributes']

print(data.dtype.names)
print(attr['q'])

header = []
csv = []
fmt = ""
cnt = 0;
for name in data.dtype.names:
  if data[name].ndim == 1:
    if name in attr.dtype.names and attr[name]['name'].astype('U32') != "":
      header.append(attr[name]['name'].astype('U32'))
    else:
      print(name)
      header.append(name)
    if isinstance(data[name][0], np.string_):
      data_to_add = data[name][:].astype('U32')
      fmt += "%s,"
    else:
      data_to_add = data[name][:]
      fmt += "%.18e,"
    if csv == []:
      csv = data_to_add
    else:
      csv = np.vstack([csv, data_to_add])
    cnt+=1
  else:
    for i in range(np.shape(data[name])[1]):
      if name in attr.dtype.names and attr[name][0,i]['name'].astype('U32') != "":
        header.append(attr[name][0,i]['name'].astype('U32'))
      else:
        print(name)
        header.append(name + str(i))
      if isinstance(data[name][0,1], np.string_):
        #data_str = value.decode('ascii')
        data_to_add = data[name][:,i].tostring().decode('ascii')
        fmt += "b%s,"
      else:
        data_to_add = data[name][:,i]
        fmt += "%.18e,"
      if csv == []:
        csv = data_to_add
      else:
        csv = np.vstack([csv, data_to_add])
      cnt+=1

#print(cnt)
csv = np.transpose(csv)
#print(csv)
header_str = ""
print(header)
separator = ';'
header_str = separator.join(header)
print(header_str)


out_filename = os.path.splitext(filename)[0] +".txt"
np.savetxt(out_filename, csv, delimiter=';', fmt="%s", header = header_str )
