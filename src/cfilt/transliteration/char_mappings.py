#Copyright Anoop Kunchukuttan 2015 - present
# 
#This file is part of the IITB Unsupervised Transliterator 
#
#IITB Unsupervised Transliterator is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#IITB Unsupervised Transliterator  is distributed in the hope that it will be useful, 
#but WITHOUT ANY WARRANTY; without even the implied warranty of 
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
#GNU General Public License for more details. 
#
#You should have received a copy of the GNU General Public License 
#along with IITB Unsupervised Transliterator.   If not, see <http://www.gnu.org/licenses/>.

from collections import defaultdict
import pprint 

en_il_rules={
    0x01: ['n','a'],
    0x02: ['m','n'],
    0x03: ['h'],
    0x05: ['a','u','o','i','ea','ou|!S','e|!S'],
    0x06: ['a','aa','ah','o','e','aw','i','ho'],
    0x07: ['i','e','ai|!S|AC'],
    0x08: ['i','e','ee','ea','ei','ey','ie','y','eigh','ai|!S|AC'],
    0x09: ['u','oo','ow|!S|AC'],
    0x0a: ['u','oo','ow|!S|AC'],
    0x0b: ['ri','ru','hri'],
    0x0c: ['lru'],
    0x0d: ['a','e','eigh','aye|!S|AC','ay|!S|AC'],
    0x0e: ['a','e','eigh','aye|!S|AC','ay|!S|AC'],
    0x0f: ['a','e','eigh','ai','aye|!S|AC','ay|!S|AC'],
    0x10: ['a','e','eigh','ai','aye|!S|AC','ay|!S|AC'],
    0x11: ['o','au','ho','aw','a','ow'],
    0x12: ['au','o'],
    0x13: ['o','oa','ow'],
    0x14: ['au','ow','ou'],
    0x15: ['k','c','q','ck|!S','ch','lk|!S|AV'],
    0x16: ['kh','ck','k'],
    0x17: ['g','gh'],
    0x18: ['gh','g'],
    0x19: ['n'],
    0x1a: ['ch','tu|!S','tch|!S','cz'],
    0x1b: ['chh'],
    0x1c: ['j','g','z','s|!S','c|!S','zh','dge|!S','x|S','d|!S','ssio|!S','tio|!S','sio|!S|!E'],
    0x1d: ['z','jh','s','zh'],
    0x1e: ['ny','n'],
    0x1f: ['t','bt|!S','th'],
    0x20: ['th','t'],
    0x21: ['d','ld|!S|AV','r|!S','rh|E'],
    0x22: ['dh','rh|E','d'],
    0x23: ['n','kn','pn|S'],
    0x24: ['t','th','gth|!S'],
    0x25: ['th','t','gth|!S'],
    0x26: ['d','th'],
    0x27: ['dh','d'],
    0x28: ['n','kn','pn|S','ln|!S','gn|!S|AV'],
    0x2a: ['p','pp|!S','ph'],
    0x2b: ['gh|!S','f','ph','ff|!S','lf|!S|AV'],
    0x2c: ['b'],
    0x2d: ['bh'],
    0x2e: ['m','lm|!S','mn|!S'],
    0x2f: ['e','i','y','u'],
    0x30: ['r','wr|!E','rh|E'],
    0x31: ['r','wr|!E','rh|E'],
    0x32: ['l','tl|!S|AC'],
    0x33: ['l','tl|!S|AC'],
    0x35: ['v','w','wh|!E','u|!S','o|!S'],
    0x36: ['sh','c','tio|!S|!E','ssio|!S|!E','ch','sch','s','t'],
    0x37: ['sh','c','tio|!S|!E','ssio|!S|!E','ch','sch','s','t'],
    0x38: ['s','c','sc|!E','ps|S','ts|S','sh','z'],
    0x39: ['h','wh|S','gh'],
    0x3e: ['a','aa','ah','o','aw','augh','ough'],
    0x3f: ['i','e','y'],
    0x40: ['i','e','ee','ea','eh','ei','ie','y','ey','eigh'],
    0x41: ['u','oo','ew','ue','oe','ou'],
    0x42: ['u','oo','ew','ue','oe','ou','ui'],
    0x43: ['ri','ru','roo'],
    0x45: ['a','e','ae'],
    0x47: ['e','a','eigh','ay','ai','ei','ea','ey','ae'],
    0x48: ['ai','igh','y','a','i','ei','ie','u','o','e'],
    0x49: ['au','o','a','aw','ow','oh','ough','augh','oa'],
    0x4b: ['o','oa','ow','ough','ou','eau','oe'],
    0x4c: ['au','ow','ou','o'],
    0x58: ['k','c','q'],
    0x59: ['kh'],
    0x5a: ['g'],
    0x5b: ['j','g','z','s|!S','c|!S','ssio|!S|!E','tio|!S|!E','sio|!S|!E'],
    0x5c: ['d','rh|!S','r'],
    0x5d: ['dh','rh|!S'],
    0x5e: ['ph','f','lf|!S|AV'],
    0x5f: ['y'],
}

def remove_constraint_rules(mapping_rules): 

    new_mapping_rules=defaultdict(list)

    for k,v in mapping_rules.iteritems(): 
        v_new=filter(lambda x: u'|' not in x  ,v)
        if len(v_new)>0:
            new_mapping_rules[k]=v_new
    
    return new_mapping_rules

def remove_constraints(mapping_rules): 

    new_mapping_rules=defaultdict(list)

    for k,v in mapping_rules.iteritems(): 
        v_new=map(lambda x: x.split('|')[0]  ,v)
        if len(v_new)>0:
            new_mapping_rules[k]=v_new
    
    return new_mapping_rules

def filter_by_val_len(mapping_rules,max_len=2): 

    new_mapping_rules=defaultdict(list)

    for k,v in mapping_rules.iteritems(): 
        v_new=filter( lambda x: len(x.split('|')[0])<=max_len  , v )
        if len(v_new)>0:
            new_mapping_rules[k]=v_new

    return new_mapping_rules

def conv_upper(mapping_rules):
    """
    Assumes there are no constraints
    """

    new_mapping_rules=defaultdict(list)

    for k,v in mapping_rules.iteritems(): 
        v_new=map(lambda x: x.upper()  ,v)
        if len(v_new)>0:
            new_mapping_rules[k]=v_new

    return new_mapping_rules
