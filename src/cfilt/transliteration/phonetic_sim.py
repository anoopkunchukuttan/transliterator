from indicnlp.script.indic_scripts import * 

def sim1(v1,v2,base=5.0): 

    dotprod=float(np.dot( v1, v2.T ))
    return np.power(base,dotprod) 

#def sim2(v1,v2,base=5.0): 
#
#    ## Weight vector
#    # phonetic_weight_vector=np.array([
#    #     60.0,60.0,60.0,60.0,60.0,60.0,
#    #     1.0,1.0,
#    #     30.0,30.0,30.0,
#    #     40.0,40.0,
#    #     50.0,50.0,50.0,50.0,50.0,
#    #     40.0,40.0,40.0,40.0,40.0,
#    #     5.0,5.0,
#    #     10.0,10.0,
#    #     10.0,10.0,
#    # ])
#    
#    phonetic_weight_vector=np.array([
#        #6.0,6.0,6.0,6.0,6.0,6.0,
#        0.01,0.01,0.01,0.01,0.01,0.01,
#        0.1,0.1,
#        3.0,3.0,3.0,
#        4.0,4.0,
#        5.0,5.0,5.0,5.0,5.0,
#        4.0,4.0,4.0,4.0,4.0,
#        0.5,0.5,
#        1.0,1.0,
#        1.0,1.0,
#    ])
#
#    v1_weighted=np.multiply(v1,phonetic_weight_vector)
#    dotprod=float(np.dot( v1_weighted, v2.T ))
#    return np.power(base,dotprod) 

#def simx(v1,v2,base=5.0): 
#    #dotprod=float(np.dot( v1, v2 ))
#    #cos_sim=dotprod/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))
#    #return cos_sim

def accumulate_vectors(v1,v2): 
    """
    not a commutative operation
    """

    if is_consonant(v1) and is_halant(v2): 
        v1[PVIDX_BT_HALANT]=1
        return v1
    elif is_consonant(v1) and is_nukta(v2): 
        v1[PVIDX_BT_NUKTA]=1
        return v1
    elif is_consonant(v1) and is_dependent_vowel(v2): 
        return or_vectors(v1,v2)
    elif is_anusvaar(v1) and is_consonant(v2): 
        return or_vectors(v1,v2)
    else: 
        return invalid_vector()

