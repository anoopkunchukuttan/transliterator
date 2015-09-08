from indicnlp.script.indic_scripts import * 

def sim1(v1,v2,base=5.0): 

    dotprod=float(np.dot( v1, v2 ))
    return np.power(base,dotprod) 

    #dotprod=float(np.dot( v1, v2 ))
    #cos_sim=dotprod/(np.sqrt(np.dot(v1,v1))*np.sqrt(np.dot(v2,v2)))
    #return cos_sim

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

