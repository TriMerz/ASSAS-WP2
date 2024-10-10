#!/usr/bin/env python3
# favorite_queue = gpu
import os
from os.path import join, isfile, isdir
import sys
import numpy as np
import astec
import tensorflow as tf
models=os.path.join('..','sg-metamodel')
sys.path.append(models)
sys.path.append(os.path.join('..','fluent'))

import fluent.path
import gv_model as gvm
import gv_metamodel as gvtm
import joblib
# from astools import *

import pyastec as pyas
pymdb = pyas.mdb
from WithModulesAstecMain import WithModulesAstecMain

# Could be read
n_steps=5 # "Normal" steps
n_meta_steps=20 # "Meta" steps

def die(msg):
    print("An error occured in script: "+__file__+": "+msg)
    sys.exit(1)
#
def display_family_names(base):
    pyas.odprint(base)
    for i in range(1-pyas.odessa_shift(), pyas.odbase_family_number(base)+1-pyas.odessa_shift()):
        print(pyas.odbase_name(base,i))
_SEP=':'

#
class Hal(WithModulesAstecMain):

    def __init__(self, model_path:str, mdat:str):
        super().__init__(mdat)

        self.metadata = gvm.load_gv_model_metadata(join(model_path,'metadata.pkl'))
        self.scaler = joblib.load(join(model_path, 'scaler' ))

        self.gv_adapters = {gv:gvtm.WindowAdapter(gvtm.WINDOW_SIZE, gvtm.INPUT_WIDTH) for gv in gvm.GVS}
        gv_paths_and_metadata = {gv:gvm.generate_gv_paths(gv, self.metadata.name_mapping) for gv in gvm.GVS}
        self.gv_input_paths={}
        self.gv_output_paths={}
        for gv in gvm.GVS:
            input=list()
            output=list()
            for (gv_path, metadata) in gv_paths_and_metadata[gv]:
                if metadata.input:input.append(gv_path)
                if metadata.output:output.append(gv_path)
            self.gv_input_paths[gv] = input
            self.gv_output_paths[gv] = output
        # self.gv_input_paths = {gv:[gv_path for (gv_path, metadata) in gv_paths_and_metadata[gv] if metadata.input] for gv in gvm.GVS}
        # self.gv_output_paths = {gv:[gv_path for (gv_path, metadata) in gv_paths_and_metadata[gv] if metadata.output] for gv in gvm.GVS}
        self.model = tf.keras.models.load_model(join(model_path,'sg_metamodel'))
        self.model.summary()

        self.iter = 0
        self.NB_META_STEPS = 10
        self.first = True
        self.logfile = open('Hal.lst','w')
        self.meta_conectis=[]

        for i in range(1,5):            
            self.meta_conectis.append( "QST"+str(i) )
            self.meta_conectis.append( "QHB"+str(i) )
            self.meta_conectis.append( "QCB"+str(i) )

        self.tagged = False
            
    def log(self, msg):
      self.logfile.write(msg+"\n")
      self.logfile.flush()

    def _astec_module(self, success, module_name, tbeg, tend):
        super()._astec_module(success, module_name, tbeg, tend)

# Interaction with base

    def compute_cesar_icare(self):
        self.collect_data()
        if( self.iter < n_steps ) :
            self.log("Running CESAR-ICARE modules at time "+str(self.time_beg))
            for conn in self.meta_conectis:
                self.close_connecti(conn)
            if self.iter==0:
                self.force_restart()
                pyas.raise_database_has_changed()
            self._astec_module(self.success, "CESAR_ICARE", self.tbeg, self.tend)
            self.meta_iter = 0
        else:
            self.log("Running meta-simulator at time "+str(self.time_beg) )
            self.meta_step()
        self.iter = self.iter + 1

    def get_index(self, circuit, family, name):
        result = -1
        for i in range(1-pyas.odessa_shift(), pyas.odbase_size( circuit, family )+1-pyas.odessa_shift()):
            sub=pyas.odbase_get_odbase( circuit, family, i )
            if( pyas.odbase_size( sub, "NAME" ) > 0 ):
                a_name = pyas.odbase_get_string( sub, "NAME", 1-pyas.odessa_shift() )
                if( a_name == name ) :
                    result = i
                    break
        if( result == -1 ):
            die("Unable to find "+name+" within family "+family)
        return result
    
    def tag(self, circuit_name, family, name, dummy_arg):
        if circuit_name is None:
            circuit=pyas.root_database()
        else:
            circuit = pyas.odbase_get_odbase( pyas.root_database(), circuit_name, 1-pyas.odessa_shift() )
        index = self.get_index( circuit, family, name )
        vol = pyas.odbase_get_odbase( circuit, family, index )
        pyas.odbase_put_int( vol, "INDEX", index, 1-pyas.odessa_shift() )
        
    def hide(self, circuit_name, family, name, dummy_arg):
        if circuit_name is None:
            circuit=pyas.root_database()
        else:
            circuit = pyas.odbase_get_odbase( pyas.root_database(), circuit_name, 1-pyas.odessa_shift() )
        index = self.get_index( circuit, family, name )
        vol = pyas.odbase_get_odbase( circuit, family, index )
        new_index = pyas.odbase_size( circuit, "H"+family )
        pyas.odbase_insert_odbase(circuit,"H"+family, vol, new_index+1-pyas.odessa_shift())
        pyas.odbase_delete_element(circuit,family, index)
        self.log( "Hiding "+family+" named "+name)
        
    def hide_link_in_family(self, circuit, family, link, name):
        for i in range(1-pyas.odessa_shift(), pyas.odbase_size( circuit, family )+1-pyas.odessa_shift() ):
            vol = pyas.odbase_get_odbase( circuit, family, i )
            for j in range(1-pyas.odessa_shift(), pyas.odbase_size( vol, link )+1-pyas.odessa_shift() ):
                jun = pyas.odbase_get_string( vol, link, j )
                if( jun == name ):
                    pyas.odbase_delete_element(vol, link, j)
                    pyas.odbase_insert_string(vol, "H"+link, name, 1-pyas.odessa_shift())
                    self.log( "Hiding in "+family+" link " +link+" to "+name)
        
    def restore(self, circuit, family):
        N = pyas.odbase_size( circuit, "H"+family )
        indexing=list()
        for i in range(1-pyas.odessa_shift(), N+1-pyas.odessa_shift() ):
            obj=pyas.odbase_get_odbase(circuit, "H"+family, i)            
            # new_index = pyas.odbase_size( circuit, family )
            new_index = pyas.odbase_get_int( obj, "INDEX", 1-pyas.odessa_shift() )
            indexing.append([new_index, obj])
        sorted_list=sorted(indexing, key=lambda pair: pair[0])
        for index,obj in sorted_list:      
            pyas.odbase_insert_odbase(circuit,family, obj, index)
            name = pyas.odbase_get_string(obj, "NAME", 1-pyas.odessa_shift())
            self.log( "Restoring "+family+" named "+name+" at "+str(index))
        for i in range( N-pyas.odessa_shift(),-pyas.odessa_shift(),-1 ):
            pyas.odbase_delete_element(circuit, "H"+family, i)

    def compute_flow( self, circuit, alter_jun_name, spec, vol_name, alter_vol_name ):

        self.log( "\n   Compute flow for "+spec )
        dico_massic_flow = { 'STEAM' : 'q_m_stea', 'WATER' : 'q_m_liq' }
        jun = pyas.odbase_get_odbase( circuit, "JUNCTION", self.get_index( circuit, "JUNCTION", alter_jun_name ) )
        jun_geom = pyas.odbase_get_odbase( jun, "GEOM", 1-pyas.odessa_shift() )
        S = pyas.odbase_get_double( jun_geom, "S", 1-pyas.odessa_shift() )
        if(spec=='WATER'):
            post="liq"
        else:
            post="gas"
        jun_ther = pyas.odbase_get_odbase( jun, "THER", 1-pyas.odessa_shift() )
        Qr1 = pyas.odbase_get_odr1( jun_ther, dico_massic_flow[spec], 1-pyas.odessa_shift() )
        Q = pyas.odr1_get( Qr1, 2-pyas.odessa_shift() )
        upstream_vol = alter_vol_name
        if( Q < 0. ) : upstream_vol = vol_name
        donnor_vol = pyas.odbase_get_odbase( circuit, "VOLUME", self.get_index( circuit, "VOLUME", upstream_vol )  )
        donnor_vol_ther = pyas.odbase_get_odbase( donnor_vol, "THER", 1-pyas.odessa_shift() )
        Tr1 = pyas.odbase_get_odr1( donnor_vol_ther, "T_"+post, 1-pyas.odessa_shift() )
        T = pyas.odr1_get( Tr1, 2-pyas.odessa_shift() )
        Pr1 = pyas.odbase_get_odr1( donnor_vol_ther, "P", 1-pyas.odessa_shift() )
        P = pyas.odr1_get( Pr1, 2-pyas.odessa_shift() )

        return [ Q, T, P ]

    def get( self, base, family:str, name:str ):
        i = self.get_index( base, family, name )
        return pyas.odbase_get_odbase( base,  family, i )

    def transfer_from_vol_ther_to_connecti_ther( self, volume_ther, connecti_ther, key:str ):

        tab = pyas.odbase_get_odr1( volume_ther, key, 1-pyas.odessa_shift() )
        pyas.odbase_put_odr1( connecti_ther, key, tab, 1-pyas.odessa_shift() )
        self.log("Put "+key+" = "+str(pyas.odr1_get( tab, 2-pyas.odessa_shift() )))
    
    def open_and_feed_connecti(self, conn, init):
        

        connecti = self.get( pyas.root_database(), "CONNECTI", conn )
        pyas.odbase_put_string( connecti, 'STAT', 'ON', 1-pyas.odessa_shift() )

        vol_name = pyas.odbase_get_string( connecti, "VOLUME", 1-pyas.odessa_shift() )
        circuit_name = pyas.odbase_get_string( connecti, "TO", 1-pyas.odessa_shift() )
        if( circuit_name=='USER' ): circuit_name = pyas.odbase_get_string( connecti, "FROM", 1-pyas.odessa_shift() )
        circuit = pyas.odbase_get_odbase( pyas.root_database(), circuit_name, 1-pyas.odessa_shift() )
        
        private = pyas.odbase_get_odbase( connecti, "PRIVATE", 1-pyas.odessa_shift() )
        alter_vol_name = pyas.odbase_get_string( private, "VOLUME", 1-pyas.odessa_shift() )
        alter_jun_name = pyas.odbase_get_string( private, "JUNCTION", 1-pyas.odessa_shift() )
        connecti_type = pyas.odbase_get_string( connecti, "TYPE", 1-pyas.odessa_shift() )

        self.log( "\n Open and feed "+conn+" of type "+ connecti_type )
        
        # Connecti of type SOURCE
        if( connecti_type=="SOURCE" ):
                for i in range(1-pyas.odessa_shift(), pyas.odbase_size( connecti, "SOURCE" )+1-pyas.odessa_shift() ):
                    source = pyas.odbase_get_odbase( connecti, "SOURCE", i )
                    spec = pyas.odbase_get_string( source, "SPEC", 1-pyas.odessa_shift() )
                    Q, T, P = self.compute_flow( circuit, alter_jun_name, spec, vol_name, alter_vol_name )
                    if init:
                        tab = pyas.odr1_init()
                        pyas.odr1_put( tab, 1-pyas.odessa_shift(), self.time_beg )
                        pyas.odr1_put( tab, 2-pyas.odessa_shift(), Q )
                        pyas.odr1_put( tab, 3-pyas.odessa_shift(), T )
                        pyas.odr1_put( tab, 4-pyas.odessa_shift(), P )
                        pyas.odbase_put_odr1( source, "FLOW", tab, 1-pyas.odessa_shift() )
                    else:
                        tab = pyas.odbase_get_odr1( source, "FLOW", 1-pyas.odessa_shift() )
                        pyas.odr1_put( tab, 5-pyas.odessa_shift(), self.time_end )
                        pyas.odr1_put( tab, 6-pyas.odessa_shift(), Q )
                        pyas.odr1_put( tab, 7-pyas.odessa_shift(), T )
                        pyas.odr1_put( tab, 8-pyas.odessa_shift(), P )
                    pyas.odbase_put_odr1( source, "FLOW", tab, 1-pyas.odessa_shift() )
                    
        elif( connecti_type=="BCPRESS" ):
            junction = self.get( circuit, "JUNCTION", alter_jun_name )
            junction_ther = pyas.odbase_get_odbase( junction, "THER", 1-pyas.odessa_shift() )
            volume = self.get( circuit, "VOLUME", alter_vol_name )
            volume_ther = pyas.odbase_get_odbase( volume, "THER", 1-pyas.odessa_shift() )
            connecti_ther = pyas.odbase_get_odbase( connecti, "THER", 1-pyas.odessa_shift() )
            
            #for key in [ 'P', 'P_h2', 'P_n2', 'P_steam', 'T_liq', 'T_gas', 'x_alfa' ]:
            # Problem occurs with P_n2 ...
            for key in [ 'P', 'P_h2', 'P_steam', 'T_liq', 'T_gas', 'x_alfa' ] :#, 'P_h2', 'P_n2', 'P_steam' ]:
                self.transfer_from_vol_ther_to_connecti_ther( volume_ther, connecti_ther, key )
            
        else:
            die("Unknown connecti type "+connecti_type)    
                    
    def transfer_from_connecti_ther_to_junction_ther( self, connecti_ther, junction_ther, key ):

        tab = pyas.odbase_get_odr1( connecti_ther, key, 1-pyas.odessa_shift() )
        pyas.odbase_put_odr1( junction_ther, key, tab, 1-pyas.odessa_shift() )
        self.log("Get "+key+" = "+str(pyas.odr1_get( tab, 2-pyas.odessa_shift() )))
        
    def close_connecti(self, conn):
        
        self.log( "\n Close "+conn )
        connecti = self.get( pyas.root_database(), "CONNECTI", conn )
        pyas.odbase_put_string( connecti, 'STAT', 'OFF', 1-pyas.odessa_shift() )
        connecti_type = pyas.odbase_get_string( connecti, "TYPE", 1-pyas.odessa_shift() )
            
    def report_connecti_to_junction(self, conn):
        
        self.log( "\n Close "+conn )
        connecti = self.get( pyas.root_database(), "CONNECTI", conn )
        connecti_type = pyas.odbase_get_string( connecti, "TYPE", 1-pyas.odessa_shift() )

        if( connecti_type=="BCPRESS" ):
            circuit_name = pyas.odbase_get_string( connecti, "TO", 1-pyas.odessa_shift() )
            if( circuit_name=='USER' ): circuit_name = pyas.odbase_get_string( connecti, "FROM", 1-pyas.odessa_shift() )
            circuit = pyas.odbase_get_odbase( pyas.root_database(), circuit_name, 1-pyas.odessa_shift() )
            private = pyas.odbase_get_odbase( connecti, "PRIVATE", 1-pyas.odessa_shift() )
            connecti_ther = pyas.odbase_get_odbase( connecti, "THER", 1-pyas.odessa_shift() )
            alter_jun_name = pyas.odbase_get_string( private, "JUNCTION", 1-pyas.odessa_shift() )
            junction = self.get( circuit, "JUNCTION", alter_jun_name )
            junction_ther = pyas.odbase_get_odbase( junction, "THER", 1-pyas.odessa_shift() )
            
            for key in [ 'v_liq', 'v_gas' ]:
                self.transfer_from_connecti_ther_to_junction_ther( connecti_ther, junction_ther, key )
            
    def prepare_cesar_computation(self):
        
        self.log( "\nPreparing CESAR calculation" )

        self.log( "\n Initialize connectis" )
        for conn in self.meta_conectis:
            self.open_and_feed_connecti(conn, True)

        self.compute_meta_model()
        
        self.log( "\n Feed connectis" )
        for conn in self.meta_conectis:
            self.open_and_feed_connecti(conn, False)

        self.hide_metamodelled_part_of_circuit()

        
    def hide_metamodelled_part_of_circuit(self):
        self.log( "\n Hiding objects modelled by meta-model" )

        if not self.tagged:
            self.apply_to_metamodelled_part('tag',None)
            self.tagged = True
        self.apply_to_metamodelled_part('hide',None)
        
    def apply_to_metamodelled_part(self,method_name,arg):
        method=getattr(self,method_name)
        for gv in range(1,5):
                # Move primary volumes, junction and walls of interest
                primary="PRIMARY"
                for vol in gvm.generate_primary_volume_datas(gv):
                    method( primary, "VOLUME", vol.name, arg )
                for jun in gvm.generate_primary_junction_datas(gv):
                    method( primary, "JUNCTION", jun.name, arg )
                   # self.hide_link_in_family(primary,"VOLUME","JUNCTION",jun.name)
                for w in gvm.generate_primary_wall_datas(gv):
                    method( primary, "WALL", w.name, arg )

                # Move secondary volumes, junction and walls of interest
                secondary = "SECONDAR"
                for vol in gvm.generate_secondar_volume_datas(gv):
                    method( secondary, "VOLUME", vol.name, arg )
                for jun in gvm.generate_secondar_junction_datas(gv):
                    method( secondary, "JUNCTION", jun.name, arg )
                   # self.hide_link_in_family(secondary,"VOLUME","JUNCTION",jun.name)
                for w in gvm.generate_secondar_wall_datas(gv):
                    method( secondary, "WALL", w.name, arg )

                # Move connectis of interest
                for conn in gvm.generate_connecti_datas(gv):
                    method( None, "CONNECTI", conn.name, arg )
                     
    def extract_data(self, source, dest ):

        for i in range(1-pyas.odessa_shift(), pyas.odbase_family_number(source)+1-pyas.odessa_shift() ):
            fname = pyas.odbase_name(source,i).strip()
            ftyp = pyas.odbase_type(source,fname)
            fnum = pyas.odbase_size(source,fname)

            for i in range(1-pyas.odessa_shift(), fnum+1-pyas.odessa_shift()):
                    if ( ftyp==pyas.od_base ):
                        sub_source=pyas.odbase_get_odbase( source, fname, i )
                        sub_dest=pyas.odbase_get_odbase( dest, fname, i )
                        self.extract_data(sub_source, sub_dest)
                    elif( ftyp==pyas.od_r0 ):
                        pyas.odbase_put_double(dest,fname,pyas.odbase_get_double(source,fname,i),i)
                    elif( ftyp==pyas.od_i0 ):
                        pyas.odbase_put_int(dest,fname,pyas.odbase_get_int(source,fname,i),i)
                    elif( ftyp==pyas.od_r1 ):
                        pyas.odbase_put_odr1(dest,fname,pyas.odbase_get_odr1(source,fname,i),i)
                    elif( ftyp==pyas.od_c0 ):
                        pyas.odbase_put_string(dest,fname,pyas.odbase_get_string(source,fname,i),i)
                    elif( ftyp==pyas.od_t ):
                        pyas.odbase_put_odt(dest,fname,pyas.odbase_get_odt(source,fname,i),i)
                    else:
                        die("Unsupported type "+str(ftyp))
                        
            
    def extract(self, circuit_name, family, name, reduced):
        self.log("Extract "+name+" in family "+family+" of circuit "+str(circuit_name))
        if circuit_name is None:
            circuit=pyas.root_database()
            reduced_circuit = reduced
        else:
            circuit = pyas.odbase_get_odbase( pyas.root_database(), circuit_name, 1-pyas.odessa_shift() )
            reduced_circuit = pyas.odbase_get_odbase( reduced, circuit_name, 1-pyas.odessa_shift() )
        vol = pyas.odbase_get_odbase( circuit, family, self.get_index( circuit, family, name ) )
        reduced_vol = pyas.odbase_get_odbase( reduced_circuit, family, self.get_index( reduced_circuit, family, name ) )
        self.extract_data(reduced_vol,vol)

    def put_values(self,base,time,values):
        for v in values:
            tab=pyas.odr1_init()
            pyas.odr1_put(tab,1-pyas.odessa_shift(),time)
            pyas.odr1_put(tab,2-pyas.odessa_shift(),values[v])
            pyas.odbase_put_odr1(base,v,tab,1-pyas.odessa_shift() )

    def get_values(self,base):
        r={}
        for i in range(1-pyas.odessa_shift(), pyas.odbase_family_number(base)+1-pyas.odessa_shift() ):
            fname = pyas.odbase_name(base,i).strip()
            ftyp = pyas.odbase_type(base,fname)
            fnum = pyas.odbase_size(base,fname)
            if( ftyp==pyas.od_r0 ):
                r[fname]=pyas.odbase_get_double(base,fname,1-pyas.odessa_shift())
            elif( ftyp==pyas.od_r1 ):
                r[fname]=pyas.odr1_get( pyas.odbase_get_odr1(base,fname,1-pyas.odessa_shift()), 2-pyas.odessa_shift() )
            else:
                die("Unsupported type "+str(ftyp)+" for "+fname)
        return r
    
                
    def complete_missing_data_from_metamodelled_ones(self, circuit_name, family, name, dummy_arg):
        x = pyas.new_doublep()
        R=8.314
        pymdb.get('H2O','M',x)
        Mh2o=pyas.doublep_value(x)
        pymdb.get('H2','M',x)
        Mh2=pyas.doublep_value(x)
        self.log("Complete "+name+" in family "+family+" of circuit "+str(circuit_name))
        if circuit_name is None:
            circuit=pyas.root_database()
        else:
            circuit = pyas.odbase_get_odbase( pyas.root_database(), circuit_name, 1-pyas.odessa_shift() )
        obj = pyas.odbase_get_odbase( circuit, family, self.get_index( circuit, family, name ) )
        if family=="VOLUME":
            vol = obj
            computed={}
            geom = pyas.odbase_get_odbase( vol, "GEOM", 1-pyas.odessa_shift() )
            volume = pyas.odbase_get_double( geom, "V", 1-pyas.odessa_shift() )
            
            ther = pyas.odbase_get_odbase( vol, "THER", 1-pyas.odessa_shift() )
            v = self.get_values( ther )
            
            computed['P'] = v['P_h2']+v['P_steam']
            
            pymdb.get('H2O','P_sat(T)',v['T_liq'],x)
            computed['Psat']=pyas.doublep_value(x)
            
            pymdb.get('H2O','rho_l(T,P)',v['T_liq'],computed['P'],x)
            computed['rho_liq']=pyas.doublep_value(x)
            
            pymdb.get('H2O','T_sat(P)',v['P_steam'],x) # To check what pressure should be used
            computed['T_sat']=pyas.doublep_value(x)

            V_liq = volume * ( 1.0 - v['x_alfa'] )
            V_gas = volume *  v['x_alfa']
            
            computed['m_liq']= V_liq * computed['rho_liq']

            computed['m_steam'] = v['P_steam']*V_gas/ R / v['T_gas'] * Mh2o

            computed['rho_gas'] =V_gas/ R / v['T_gas']* ( v['P_steam'] * Mh2o + v['P_h2'] * Mh2 ) / V_gas

            computed['x_steam'] = v['P_steam'] * Mh2o / ( v['P_steam'] * Mh2o + v['P_h2'] * Mh2 )

            Vl_bar = 0.
            Vg_bar = 0.
            N = pyas.odbase_size(vol,'JUNCTION')
            for j in range(1-pyas.odessa_shift(), N+1-pyas.odessa_shift() ):
                jname = pyas.odbase_get_string(vol,'JUNCTION',j)
                junc = pyas.odbase_get_odbase( circuit, "JUNCTION", self.get_index( circuit, "JUNCTION", jname ) )
                juncther = pyas.odbase_get_odbase( junc, "THER", 1-pyas.odessa_shift() )
                Vgr1 = pyas.odbase_get_odr1( juncther, "v_gas", 1-pyas.odessa_shift() )
                Vg = pyas.odr1_get( Vgr1, 2-pyas.odessa_shift() )
                Vlr1 = pyas.odbase_get_odr1( juncther, "v_liq", 1-pyas.odessa_shift() )
                Vl = pyas.odr1_get( Vlr1, 2-pyas.odessa_shift() )
                Vl_bar = Vl_bar + Vl / N
                Vg_bar = Vg_bar + Vg / N
                
            computed['VG_bar'] = Vg_bar
            computed['VL_bar'] = Vl_bar

            # A very crude approximation
            computed['P_UP'] = v[ 'P' ]
            
            # TODO complete fm_itf
            self.log("  Complete for volume "+name+" the values "+str( computed ) )
            self.put_values( ther, self.time_end, computed )
            
        if family=="JUNCTION":
            computed={}
            jun = obj
            geom = pyas.odbase_get_odbase( jun, "GEOM", 1-pyas.odessa_shift() )
            S = pyas.odbase_get_double( geom, "S", 1-pyas.odessa_shift() )
            
            ther = pyas.odbase_get_odbase( jun, "THER", 1-pyas.odessa_shift() )
            v = self.get_values( ther )
            
            computed['delta_v'] = v['v_gas'] - v['v_liq']

            donnorg = "NV_UP"
            if( v['v_gas'] < 0. ): donnorg = "NV_DOWN"
            vol_donnorg_name = pyas.odbase_get_string( jun, donnorg, 1-pyas.odessa_shift() )
            vol_donnorg = pyas.odbase_get_odbase( circuit, "VOLUME", self.get_index( circuit, "VOLUME", vol_donnorg_name ) )
            vol_donnorg_ther =  pyas.odbase_get_odbase( vol_donnorg, "THER", 1-pyas.odessa_shift() )
            vol_donnor_v = self.get_values( vol_donnorg_ther )

            computed['q_m_gas'] = v['v_gas'] * S * vol_donnor_v[ 'rho_gas' ]
            
            donnorl = "NV_UP"
            if( v['v_liq'] < 0. ): donnorl = "NV_DOWN"
            if donnorg != donnorl:
                vol_donnorl_name = pyas.odbase_get_string( jun, donnorl, 1-pyas.odessa_shift() )
                vol_donnorl = pyas.odbase_get_odbase( circuit, "VOLUME", self.get_index( circuit, "VOLUME", vol_donnorl_name ) )
                vol_donnorl_ther =  pyas.odbase_get_odbase( vol_donnorl, "THER", 1-pyas.odessa_shift() )
                vol_donnor_v = self.get_values( vol_donnorl_ther )

            computed['q_m_liq'] = v['v_liq'] * S * vol_donnor_v[ 'rho_liq' ]
            
            self.log("  Complete for junction "+name+" the values "+str( computed ) )
            self.put_values( ther, self.time_end, computed )

            
    def odessa_value_to_array_value(self, path, value):
        if isinstance(value, str):
            if value == "OFF":
                return 0.
            elif value == "ON":
                return 1.
            else:
                raise Exception(f"Unhandled string value : {value} at path {path}")
        return value

    def normalized_value( self, gv_path, value ):
        eps = 1.0e-6
        result = value
        if isinstance(gv_path, fluent.path.R1ElementPath):
            name = gv_path.parent.name
        else:
            name = gv_path.name
            
        if name == 'x_alfa':
            if result < eps :
                result = eps
            elif result > 1.0-eps:
                result = 1.0-eps
        elif name.startswith( 'P_' ):
            if result < 0. : result = 0.
        elif name.startswith( 'T_' ):
            if result < 0. : result = 0.

        return result
    
    def compute_meta_model(self):
        
        self.log( " Computing missing part of circuit with meta-model" )

        for gv in gvm.GVS:
            adapter = self.gv_adapters[gv]
            self.log("Starting metamodel step for steam generator "+str(gv) )
            window = adapter.get_window()
            output = gvtm.metamodel_step(self.model, window, self.metadata, self.scaler )
            self.log("Ending metamodel step, computed output number "+str(len(output)))
            for index, gv_path in enumerate(self.gv_output_paths[gv]):
                value = self.normalized_value( gv_path, output[index].item() )
                self.log(f"Putting path {gv_path} "+str(output[index])+ ' norm. '+str(value))
                if isinstance(gv_path, fluent.path.R1ElementPath):
                    R1 = gv_path.parent
                    tab=gv_path.parent.get_from(pyas.root_database())
                    self.log(f"Putting value {value} at coord {gv_path.coord[0]} in tab of size "+str(pyas.odr1_size(tab)))
                    pyas.odr1_put(tab,1-pyas.odessa_shift(),self.time_end)
                    pyas.odr1_put(tab,gv_path.coord[0]+1-pyas.odessa_shift(),value)
                    R1.put_from(pyas.root_database(),tab)
                else:
                    gv_path.put_from(pyas.root_database(),value)

        self.log( "  Complete data from surogated ones" )
        self.apply_to_metamodelled_part( "complete_missing_data_from_metamodelled_ones",None ) 

    def collect_data(self):
        
        self.log( " Collect data for meta-model" )

        for gv in gvm.GVS:
            adapter = self.gv_adapters[gv]
            new_input = np.zeros(gvtm.INPUT_WIDTH, dtype=np.float32)
            for index, gv_path in enumerate(self.gv_input_paths[gv]):
                value = gv_path.get_from(pyas.root_database())
                new_input[index] = self.odessa_value_to_array_value(gv_path, value)
                self.log(f"Getting path {gv_path} "+str(new_input[index]))
            adapter.add_input(new_input)

                    
    def force_restart(self):
        restart =    pyas.odbase_init() # Force CESAR reinitialisation
        pyas.odbase_insert_odbase( pyas.root_database(), "RESTART", restart, 1-pyas.odessa_shift() )
         
    def terminate_cesar_computation(self):

        self.log( "\nTerminating CESAR calculation" )
        self.log( "\n Restoring objects modelled by meta-model" )
        # Restore primary volumes, junction and walls of interest
        primary = pyas.odbase_get_odbase( pyas.root_database(), "PRIMARY", 1-pyas.odessa_shift() )
        self.restore( primary, "VOLUME" )
        self.restore( primary, "JUNCTION" )
        self.restore( primary, "WALL" )
        
        # Restore secondary volumes, junction and walls of interest
        secondary = pyas.odbase_get_odbase( pyas.root_database(), "SECONDAR", 1-pyas.odessa_shift() )
        self.restore( secondary, "VOLUME" )
        self.restore( secondary, "JUNCTION" )
        self.restore( secondary, "WALL" )
 
        # Restore connectis of interest
        self.restore( pyas.root_database(), "CONNECTI" )
        
        for conn in self.meta_conectis:
            self.report_connecti_to_junction(conn)


# Models
    def main_loop(self):

        while self.icont > 0:
            pyas.calc_init(self.success, self.tbeg, self.tend)
            while not pyas.boolp_value(self.success):
                pyas.calc_hot_restart(self.tbeg, self.tend)
                self.time_beg=pyas.doublep_value(self.tbeg)
                self.time_end=pyas.doublep_value(self.tend)
                pyas.boolp_assign(self.success, True)
                pyas.calc_before_modules(self.first_readbase, self.tbeg, self.tend)

                self._astec_module(self.first_readbase, "READBASE"   , self.tbeg, self.tend)
                self._astec_module(self.success, "RUPUICUV"   , self.tbeg, self.tend)
                self.compute_cesar_icare()
                self._astec_module(self.success, "READBASE"   , self.tbeg, self.tend)
                self._astec_module(self.success, "DROPLET"    , self.tbeg, self.tend)
                self._astec_module(self.success, "RCSMESH"    , self.tbeg, self.tend)
                self._astec_module(self.success, "CORIUM"     , self.tbeg, self.tend)
                self._astec_module(self.success, "MEDICIS"    , self.tbeg, self.tend)
                self._astec_module(self.success, "ASCAVI"     , self.tbeg, self.tend)
                self._astec_module(self.success, "THC"        , self.tbeg, self.tend)
                self._astec_module(self.success, "CPA"        , self.tbeg, self.tend)
                self._astec_module(self.success, "PH"         , self.tbeg, self.tend)
                self._astec_module(self.success, "SOPHAEROS"  , self.tbeg, self.tend)
                self._astec_module(self.success, "SAFARI"     , self.tbeg, self.tend)
                self._astec_module(self.success, "DOSE"       , self.tbeg, self.tend)
                self._astec_module(self.success, "COVI"       , self.tbeg, self.tend)
                self._astec_module(self.success, "ISODOP"     , self.tbeg, self.tend)

                pyas.calc_after_modules(self.success, self.tbeg, self.tend)
            self.icont = pyas.tool()


    def _computation(self):
        self.main_loop()

    def meta_step(self):
        
        # Serious things start just here...
        
        self.prepare_cesar_computation()
        if self.meta_iter==0:
            self.force_restart()
            pyas.raise_database_has_changed()
        
        self._astec_module(self.success, "CESAR_ICARE", self.tbeg, self.tend)

        self.terminate_cesar_computation()

        self.meta_iter   = self.meta_iter + 1
        
        if( self.meta_iter == n_meta_steps ) :
          # Revert standard mode
          self.iter = 0


if __name__ == "__main__":
    model_path = os.path.join('..','sg-metamodel','sg_model')  # sys.argv[1]
    computation = Hal( model_path, "restart.mdat")
    # First read encoder, decoder and simulator
    # computation.read_models()
    computation.run()
    print("NORMAL END")
