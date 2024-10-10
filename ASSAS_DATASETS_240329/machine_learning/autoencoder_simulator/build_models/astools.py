"""
module containing astec algorithms
"""


import os
from os.path import *
import pyastec as pyas
import sys
from common import die
import pandas
import numpy

# Utilities
_SEP=':'


debug = False

def non_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() > 1]

dico = { "STAT" : { "ON" : 1 , "OFF" : 2, "COMPACT" : 3, "ABSENT" : 4, "DISLOCAT" : 5, "PERFORAT" : 6, "CRACKED" : 7  },
     "ZPRTNA" : { "GAS" : 1, "FLUID" : 2, "DUMMY" : 3 } ,
     "ZPHASE" : { "GAS" : 1, "FLUID" : 2, "DUMMY" : 3 } ,
     "UNIT" : { "kg-J" : 1, "kg" : 2, "kg/s-W" : 3, "kg/s" : 4, "W" : 5, "kg/s-K-P" :6  },
     "HOLS" : { "NO" : 1 , "YES" : 2 },
     "CTYPF" : { "INT" : 1 , "LOW" : 2 } } # Usefull ?
indexes = [ "NAME", "SMAT" ]
ignored = [ "DISLOCAT", "COMPACT", "ABSENT", "CRACKED", "PERFORAT" ]
ignored_families = [ "CREE" ] 
DUO = [ "T_liq", "T_gas", "x_alfa", "P_steam", "P_o2", "P_h2", "P_n2", "P_bho2", "P_co2", "v_liq", "v_gas", "FLUX" , "P" , "T_wall" , "x_alfa1" , "WTEMP", "VG", "VF", "TLIQ", "TFLU", "MASS", "PRES", "ZVELO", "VLIQ", "FPHDDRY", "FPHDWET", "FPHGAS", "FPHAERO", "FPHWATER", "ZHEWALL", "ZHEZONE", "ZHEZONS", "FDELP"]
full_arrays = [ "C", "SNGZ", "TEA", "TRTSURF", "VIEW", "DEA" ]
ignore = [ "P_vol" ]

# Parse database and return all ( path, value ) pairs
def parse( base, t, path=None ):
    sep = ':' # Path separator

    if path==None:
        path = ""
    else:
        path = path + sep

    for i in range(1-pyas.odessa_shift(), pyas.odbase_family_number(base)+1-pyas.odessa_shift() ):

        fname = pyas.odbase_name( base, i ).strip()
        fnumber = pyas.odbase_size( base, fname )
        ftype = pyas.odbase_type( base, fname )

        if fname in ignored_families: continue
        
        newpath = path + fname
        
        if debug : print("Parsing family "+newpath) 

        for j in range(1-pyas.odessa_shift(), fnumber+1-pyas.odessa_shift() ):
            
            if( ftype==pyas.od_base ):

                sub_base = pyas.odbase_get_odbase( base, fname, j )

                # Default path name
                sub_path = ""
                if( fnumber > 1 ):
                    sub_path="("+str(j)+")"
                
                # path name if NAME is given except for MATE
                for k in range(1-pyas.odessa_shift(), pyas.odbase_family_number(sub_base)+1-pyas.odessa_shift() ):
                    fn = pyas.odbase_name( sub_base, k ).strip()
                    if fn in indexes:
                        sub_path = "("
                        if( fn != "NAME" ) : sub_path = sub_path + fn + "="
                        sub_path = sub_path + pyas.odbase_get_string( sub_base, fn, 1-pyas.odessa_shift() ).strip()
                        if( fname == "MATE" ) : sub_path = sub_path + '-' + str(j)
                        sub_path = sub_path + ")"
                        break

                yield from parse( sub_base, t, newpath + sub_path )
                
            else:

              nextpath = newpath
              
              if( fnumber > 1 ):
                  nextpath = nextpath + "("+str(j)+")"
              
              if( ftype==pyas.od_r0 ):

                v = pyas.odbase_get_double( base, fname, j )
                yield ( nextpath, v )

              elif( ftype==pyas.od_i0 ):

                v = pyas.odbase_get_int( base, fname, j )
                yield ( nextpath, v )

              elif( ftype==pyas.od_r1 ):

                tab = pyas.odbase_get_odr1( base, fname, j )
                size = pyas.odr1_size( tab )
                if fname in DUO:
                  if size != 2:
                    die("DUO is not of size 2: "+nextpath)
                  yield ( nextpath + "_t[2]", pyas.odr1_get( tab, 2-pyas.odessa_shift() ) )
                   
                elif fname == "HTEM" or fname == "FLOW":
                    for k in range(1-pyas.odessa_shift(), size-pyas.odessa_shift() ):
                      yield ( nextpath + "_t[" + str(k+1) + "]", pyas.odr1_get( tab, k+1 ) )

                elif fname in full_arrays:
                  for k in range(1-pyas.odessa_shift(), size+1-pyas.odessa_shift() ):
                    yield ( nextpath + "[" + str(k) + "]", pyas.odr1_get( tab, k ) )                  

                elif not fname in ignore:
                    for k in range(1-pyas.odessa_shift(), size+1-pyas.odessa_shift()):
                      print(str(k)+" "+str(pyas.odr1_get( tab, k )))
                    die("Unable to deal with this vector "+nextpath)
                  
              elif( ftype==pyas.od_rg  ):

                tab = pyas.odbase_get_odrg( base, fname, j )
                for k in range(1-pyas.odessa_shift(), pyas.odrg_size( tab )+1-pyas.odessa_shift()):
                    key = pyas.odrg_name( tab, k )
                    yield ( nextpath + "[" +  key + "]", pyas.odrg_get( tab, key ) )
                    
              elif( ftype==pyas.od_c0  ):

                if( fname in indexes ):
                    
                    # Nothing to do
                    v = None
                    
                elif( fname in dico ):
                    v = pyas.odbase_get_string( base, fname, j )
                    if not v in dico[fname]:
                      print("Extend dictionary "+fname+" with "+v)
                      old=dico[fname]
                      old[v] = len(old)+1
                      dico[fname] = old
                      
                    if v in dico[fname] :
                       yield ( nextpath, dico[fname][v] )
                    else:
                        die("Unknown key "+v+" for " + nextpath )
                else:
                    v = pyas.odbase_get_string( base, fname, j )
                    print("Ignore unrecognized string "+fname+" = " + v + " in " + nextpath )


              else:
                if not fname in ignored:
                  die("Unsupported type "+str(ftype)+ " for " + nextpath )



def extractDataFrame(A,tstart,tend,dt):
        t = tstart
        nb=0
        mapval={}
        total_nb=int((tend-tstart)/dt)
        print('Reading '+str(total_nb)+' timesteps')
        
        # Collect each time step in a dictionnary of list
        while( t < tend ):
            print('Read '+A+' at time= '+str(t))
            try:
                full, T = read_binary_nearby( A, t )
                if abs( T - t ) > 1. :
                  raise IOError ('read_binary '+A+' read just after '+str(T)+' expected '+str(t))                 
                for ( path, value ) in parse( full, t ):
                    if(not path in mapval):
                        if debug :
                                print('Add new column '+path+' at iter '+str(nb))
                        mapval[path]=numpy.zeros(total_nb)
                    vec = mapval[path]
 
                    vec[nb]=value
                pyas.odbase_delete(full)

                nb=nb+1
                t=t+dt
                
            except Exception as e :
                print(e)
                print('Cant read after '+str(t))
                for col in mapval:
                  vec=mapval[col]
                  mapval[col]=numpy.resize(vec,nb)
                break
                
            print(' Number of columns read = '+str(len(mapval)))

        print( "Total number of rows: "+str(nb) )
        for col in mapval:
          vec=mapval[col]
          if len(vec)!=nb:
            die("For "+col+" len= "+str(len(vec))+" <> "+str(nb))
        
        return mapval

# Parse database and count all
def parse_and_count_fields( base, path=None, width="" ):
    indexes = [ "NAME", "SMAT" ]
    sep = ':' # Path separator
    
    if path==None:
        path = ""
    else:
        path = path + sep

    number = 1-pyas.odessa_shift()

    for i in range(1-pyas.odessa_shift(), pyas.odbase_family_number(base)+1-pyas.odessa_shift() ):

        fname = pyas.odbase_name( base, i ).strip()
        fnumber = pyas.odbase_size( base, fname )
        ftype = pyas.odbase_type( base, fname )

        newpath = path + fname
        subnumber = number
        for j in range(1-pyas.odessa_shift(), fnumber+1-pyas.odessa_shift() ):
            
            if( ftype==pyas.od_base ):
                
                sub_base = pyas.odbase_get_odbase( base, fname, j )
                # Default path name
                sub_path = ""
                if( fnumber > 1 ):
                    sub_path="("+str(j)+")"
                
                # path name if NAME is given
                for k in range(1-pyas.odessa_shift(), pyas.odbase_family_number(sub_base)+1-pyas.odessa_shift() ):
                    fn = pyas.odbase_name( sub_base, k ).strip()
                    if fn in indexes:
                        sub_path = "("
                        if( fn != "NAME" ) : sub_path = sub_path + fn + "="
                        sub_path = sub_path + pyas.odbase_get_string( sub_base, fn, 1-pyas.odessa_shift() ).strip() + ")"
                        break

                number = number + parse_and_count_fields( sub_base, newpath + sub_path )

            elif( ftype==pyas.od_r1 ):

                tab = pyas.odbase_get_odr1( base, fname, j )
                n = pyas.odr1_size( tab )
                number = number + n - 1
                   
            elif( ftype==pyas.od_rg  ):

                tab = pyas.odbase_get_odrg( base, fname, j )                
                number = number + pyas.odrg_size( tab )
                    
            elif( ftype==pyas.od_r0 or ftype==pyas.od_c0 or ftype==pyas.od_i0 ):

                number = number + 1
                
            elif( ftype==pyas.od_c0 ):

                string = pyas.odbase_get_string( base, fname, j )
                if not string in indexes : number = number + 1
                
            else:

                print("*********** Unsupported type "+str(ftype)+ " for " + newpath )
            
        print(newpath+" : "+str(number-subnumber))
        
    print(path+" : "+str(number)) 

    return( number )


def count_fields(A,t):
  
    full, T = read_binary_nearby( A, t )
    n=parse_and_count_fields(full)
    pyas.odbase_delete(full)
    
    return( n )
  

# Read ASTEC database saved in a binary directory
def read_binary( bin, time):
  [B,t] = read_binary_nearby( bin, time)
  if( abs(t-time) > 1.0e-1 ):
    raise IOError ('read_binary '+bin+' '+str(time)+' bad time '+str(t))

  return B

# Read ASTEC database saved in a binary directory
def read_binary_nearby( bin, time):
  if( not os.path.isdir(bin) ):
    die( 'Absent directory '+bin )
  B = pyas.odloaddir( bin, (time-1.0e-4) )
  seq = pyas.odbase_get_odbase(B,'SEQUENCE',1-pyas.odessa_shift())
  if(seq is None):
    die('read_binary_nearby '+bin+' '+str(time)+' no SEQUENCE read')
  t = pyas.odbase_get_double(seq,'TIME',1-pyas.odessa_shift())
  dt = pyas.odbase_get_double(seq,'STEP',1-pyas.odessa_shift())

  return [ B, t+dt ]

# Run testlist with ASTEC
def run_astec(testfile_name):
        # run ASTEC
        testlist=[]
        testlist += ast.get_tests(testfile_name,'.')
        ast.run_testlist(testlist)

