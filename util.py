import numpy as np
import random
import torch
import time


def list2tuple(l):
    return tuple(list2tuple(x) if type(x) == list else x for x in l)


def tuple2list(t):
    return list(tuple2list(x) if type(x) == tuple else x for x in t)


flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, tuple) else [l]


def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())


def set_global_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]:
        arg_return = eval(arg_return)
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return


def flatten_query(queries):
    """assign query structure to each sample"""
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def relation_classifier(test_queries, inverse_relations_path, symmetry_relations_path, composition_relations_path, containment_relations_path, transtivity_relations_path):
    
    with open(inverse_relations_path, 'r') as file:
        inverse_relations = []
        for line in file:
            inverse_relations.append(line.split('\n')[0])

    with open(symmetry_relations_path, 'r') as file:
        symmetry_relations = []
        for line in file:
            symmetry_relations.append(line.split('\n')[0])

    with open(composition_relations_path, 'r') as file:
        composition_relations = []
        for line in file:
            composition_relations.append(line.split('\n')[0])
    
    with open(containment_relations_path, 'r') as file:
        containment_relations = []
        for line in file:
            containment_relations.append(line.split('\n')[0])
    
    with open(transtivity_relations_path, 'r') as file:
        transtivity_relations = []
        for line in file:
            transtivity_relations.append(line.split('\n')[0])

    queries_1p = test_queries[('e', ('r',))]
    queries_2p = test_queries[('e', ('r', 'r'))]
    queries_3p = test_queries[('e', ('r', 'r', 'r'))]
    queries_2i = test_queries[(('e', ('r',)), ('e', ('r',)))]
    queries_3i = test_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))]
    queries_ip = test_queries[((('e', ('r',)), ('e', ('r',))), ('r',))]
    queries_pi = test_queries[(('e', ('r', 'r')), ('e', ('r',)))]
    queries_2in = test_queries[(('e', ('r',)), ('e', ('r', 'n')))]
    queries_3in = test_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))]
    queries_inp = test_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))]
    queries_pin = test_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))]
    queries_pni = test_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))]
    queries_2u = test_queries[(('e', ('r',)), ('e', ('r',)), ('u',))]
    queries_up = test_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))]

    ## 1p
    symmetry_queries_1p = set()
    for query in queries_1p:
        if str(query[1][0]) in symmetry_relations:
            symmetry_queries_1p.add(query)

    inverse_queries_1p = set()
    for query in queries_1p:
        if str(query[1][0]) in inverse_relations:
            inverse_queries_1p.add(query)

    composition_queries_1p = set()
    for query in queries_1p:
        if str(query[1][0]) in composition_relations:
            composition_queries_1p.add(query)
    
    containment_queries_1p = set()
    for query in queries_1p:
        if str(query[1][0]) in containment_relations:
            containment_queries_1p.add(query)

    transtivity_queries_1p = set()
    for query in queries_1p:
        if str(query[1][0]) in transtivity_relations:
            transtivity_queries_1p.add(query)

    ## 2p
    symmetry_queries_2p = set()
    for query in queries_2p:
        for r in query[1]:
            if str(r) in symmetry_relations:
                symmetry_queries_2p.add(query)
                break

    inverse_queries_2p = set()
    for query in queries_2p:
        for r in query[1]:
            if str(r) in inverse_relations:
                inverse_queries_2p.add(query)
                break

    composition_queries_2p = set()
    for query in queries_2p:
        for r in query[1]:
            if str(r) in composition_relations:
                composition_queries_2p.add(query)
                break

    containment_queries_2p = set()
    for query in queries_2p:
        for r in query[1]:
            if str(r) in containment_relations:
                containment_queries_2p.add(query)
                break
    
    transtivity_queries_2p = set()
    for query in queries_2p:
        for r in query[1]:
            if str(r) in transtivity_relations:
                transtivity_queries_2p.add(query)
                break

    ## 3p
    symmetry_queries_3p = set()
    for query in queries_3p:
        for r in query[1]:
            if str(r) in symmetry_relations:
                symmetry_queries_3p.add(query)
                break

    inverse_queries_3p = set()
    for query in queries_3p:
        for r in query[1]:
            if str(r) in inverse_relations:
                inverse_queries_3p.add(query)
                break

    composition_queries_3p = set()
    for query in queries_3p:
        for r in query[1]:
            if str(r) in composition_relations:
                composition_queries_3p.add(query)
                break

    containment_queries_3p = set()
    for query in queries_3p:
        for r in query[1]:
            if str(r) in containment_relations:
                containment_queries_3p.add(query)
                break
    
    transtivity_queries_3p = set()
    for query in queries_3p:
        for r in query[1]:
            if str(r) in transtivity_relations:
                transtivity_queries_3p.add(query)
                break

    ## 2i
    symmetry_queries_2i = set()
    for query in queries_2i:
        for p in query:
            if str(p[1][0]) in symmetry_relations:
                symmetry_queries_2i.add(query)
                break

    inverse_queries_2i = set()
    for query in queries_2i:
        for p in query:
            if str(p[1][0]) in inverse_relations:
                inverse_queries_2i.add(query)
                break

    composition_queries_2i = set()
    for query in queries_2i:
        for p in query:
            if str(p[1][0]) in composition_relations:
                composition_queries_2i.add(query)
                break

    containment_queries_2i = set()
    for query in queries_2i:
        for p in query:
            if str(p[1][0]) in containment_relations:
                containment_queries_2i.add(query)
                break

    transtivity_queries_2i = set()
    for query in queries_2i:
        for p in query:
            if str(p[1][0]) in transtivity_relations:
                transtivity_queries_2i.add(query)
                break

    ## 3i
    symmetry_queries_3i = set()
    for query in queries_3i:
        FLAG_saved = False
        for p in query:
            if str(p[1][0]) in symmetry_relations:
                symmetry_queries_3i.add(query)
                break

    inverse_queries_3i = set()
    for query in queries_3i:
        for p in query:
            if str(p[1][0]) in inverse_relations:
                inverse_queries_3i.add(query)
                break

    composition_queries_3i = set()
    for query in queries_3i:
        for p in query:
            if str(p[1][0]) in composition_relations:
                composition_queries_3i.add(query)
                break
    
    containment_queries_3i = set()
    for query in queries_3i:
        for p in query:
            if str(p[1][0]) in containment_relations:
                containment_queries_3i.add(query)
                break
    
    transtivity_queries_3i = set()
    for query in queries_3i:
        for p in query:
            if str(p[1][0]) in transtivity_relations:
                transtivity_queries_3i.add(query)
                break

    ## ip ((('e', ('r',)), ('e', ('r',))), ('r',))
    symmetry_queries_ip = set()
    for query in queries_ip:
        FLAG_saved = False
        if str(query[1][0]) in symmetry_relations:
            symmetry_queries_ip.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in symmetry_relations:
                    symmetry_queries_ip.add(query)
                    break


    inverse_queries_ip = set()
    for query in queries_ip:
        FLAG_saved = False
        if str(query[1][0]) in inverse_relations:
            inverse_queries_ip.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in inverse_relations:
                    inverse_queries_ip.add(query)
                    break

    composition_queries_ip = set()
    for query in queries_ip:
        FLAG_saved = False
        if str(query[1][0]) in composition_relations:
            composition_queries_ip.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in composition_relations:
                    composition_queries_ip.add(query)
                    break

    containment_queries_ip = set()
    for query in queries_ip:
        FLAG_saved = False
        if str(query[1][0]) in containment_relations:
            containment_queries_ip.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in containment_relations:
                    containment_queries_ip.add(query)
                    break
    
    transtivity_queries_ip = set()
    for query in queries_ip:
        FLAG_saved = False
        if str(query[1][0]) in transtivity_relations:
            transtivity_queries_ip.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in transtivity_relations:
                    transtivity_queries_ip.add(query)
                    break

    ## pi (('e', ('r', 'r')), ('e', ('r',)))
    symmetry_queries_pi = set()
    for query in queries_pi:
        FLAG_saved = False
        for p in query:
            for r in p[1]:
                if str(r) in symmetry_relations and (not FLAG_saved):
                    symmetry_queries_pi.add(query)
                    FLAG_saved = True
                    break

    inverse_queries_pi = set()
    for query in queries_pi:
        FLAG_saved = False
        for p in query:
            for r in p[1]:
                if str(r) in inverse_relations and (not FLAG_saved):
                    inverse_queries_pi.add(query)
                    FLAG_saved = True
                    break

    composition_queries_pi = set()
    for query in queries_pi:
        FLAG_saved = False
        for p in query:
            for r in p[1]:
                if str(r) in composition_relations and (not FLAG_saved):
                    composition_queries_pi.add(query)
                    FLAG_saved = True
                    break

    containment_queries_pi = set()
    for query in queries_pi:
        FLAG_saved = False
        for p in query:
            for r in p[1]:
                if str(r) in containment_relations and (not FLAG_saved):
                    containment_queries_pi.add(query)
                    FLAG_saved = True
                    break

    transtivity_queries_pi = set()
    for query in queries_pi:
        FLAG_saved = False
        for p in query:
            for r in p[1]:
                if str(r) in transtivity_relations and (not FLAG_saved):
                    transtivity_queries_pi.add(query)
                    FLAG_saved = True
                    break

    ## 2in (('e', ('r',)), ('e', ('r', 'n')))
    symmetry_queries_2in = set()
    for query in queries_2in:
        FLAG_saved = False
        for p in query:
            if str(p[1][0]) in symmetry_relations:
                symmetry_queries_2in.add(query)
                FLAG_saved = True
                break

    inverse_queries_2in = set()
    for query in queries_2in:
        for p in query:
            if str(p[1][0]) in inverse_relations:
                inverse_queries_2in.add(query)
                break

    composition_queries_2in = set()
    for query in queries_2in:
        for p in query:
            if str(p[1][0]) in composition_relations:
                composition_queries_2in.add(query)
                break

    containment_queries_2in = set()
    for query in queries_2in:
        for p in query:
            if str(p[1][0]) in containment_relations:
                containment_queries_2in.add(query)
                break

    transtivity_queries_2in = set()
    for query in queries_2in:
        for p in query:
            if str(p[1][0]) in transtivity_relations:
                transtivity_queries_2in.add(query)
                break

    ## 3in
    symmetry_queries_3in = set()
    for query in queries_3in:
        FLAG_saved = False
        for p in query:
            if str(p[1][0]) in symmetry_relations:
                symmetry_queries_3in.add(query)
                FLAG_saved = True
                break

    inverse_queries_3in = set()
    for query in queries_3in:
        for p in query:
            if str(p[1][0]) in inverse_relations:
                inverse_queries_3in.add(query)
                break

    composition_queries_3in = set()
    for query in queries_3in:
        for p in query:
            if str(p[1][0]) in composition_relations:
                composition_queries_3in.add(query)
                break

    containment_queries_3in = set()
    for query in queries_3in:
        for p in query:
            if str(p[1][0]) in containment_relations:
                containment_queries_3in.add(query)
                break
    
    transtivity_queries_3in = set()
    for query in queries_3in:
        for p in query:
            if str(p[1][0]) in transtivity_relations:
                transtivity_queries_3in.add(query)
                break

    ## inp ((('e', ('r',)), ('e', ('r','n'))), ('r',))
    symmetry_queries_inp = set()
    for query in queries_inp:
        FLAG_saved = False
        if str(query[1][0]) in symmetry_relations:
            symmetry_queries_inp.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in symmetry_relations:
                    symmetry_queries_inp.add(query)
                    FLAG_saved = True
                    break

    inverse_queries_inp = set()
    for query in queries_inp:
        FLAG_saved = False
        if str(query[1][0]) in inverse_relations:
            inverse_queries_inp.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in inverse_relations:
                    inverse_queries_inp.add(query)
                    break

    composition_queries_inp = set()
    for query in queries_inp:
        FLAG_saved = False
        if str(query[1][0]) in composition_relations:
            composition_queries_inp.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in composition_relations:
                    composition_queries_inp.add(query)
                    break
    
    containment_queries_inp = set()
    for query in queries_inp:
        FLAG_saved = False
        if str(query[1][0]) in containment_relations:
            containment_queries_inp.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in containment_relations:
                    containment_queries_inp.add(query)
                    break

    transtivity_queries_inp = set()
    for query in queries_inp:
        FLAG_saved = False
        if str(query[1][0]) in transtivity_relations:
            transtivity_queries_inp.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0]:
                if str(p[1][0]) in transtivity_relations:
                    transtivity_queries_inp.add(query)
                    break

    ## pin (('e', ('r', 'r')), ('e', ('r', 'n')))
    symmetry_queries_pin = set()
    for query in queries_pin:
        FLAG_saved = False
        if str(query[0][1][0]) in symmetry_relations or str(query[0][1][1]) in symmetry_relations:
            symmetry_queries_pin.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in symmetry_relations and (not FLAG_saved):
            symmetry_queries_pin.add(query)

    inverse_queries_pin = set()
    for query in queries_pin:
        FLAG_saved = False
        if str(query[0][1][0]) in inverse_relations or str(query[0][1][1]) in inverse_relations:
            inverse_queries_pin.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in inverse_relations and (not FLAG_saved):
            inverse_queries_pin.add(query)

    composition_queries_pin = set()
    for query in queries_pin:
        FLAG_saved = False
        if str(query[0][1][0]) in composition_relations or str(query[0][1][1]) in composition_relations:
            composition_queries_pin.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in composition_relations and (not FLAG_saved):
            composition_queries_pin.add(query)

    containment_queries_pin = set()
    for query in queries_pin:
        FLAG_saved = False
        if str(query[0][1][0]) in containment_relations or str(query[0][1][1]) in containment_relations:
            containment_queries_pin.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in containment_relations and (not FLAG_saved):
            containment_queries_pin.add(query)
    
    transtivity_queries_pin = set()
    for query in queries_pin:
        FLAG_saved = False
        if str(query[0][1][0]) in transtivity_relations or str(query[0][1][1]) in transtivity_relations:
            transtivity_queries_pin.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in transtivity_relations and (not FLAG_saved):
            transtivity_queries_pin.add(query)

    ## pni (('e', ('r', 'r', 'n')), ('e', ('r',)))
    symmetry_queries_pni = set()
    for query in queries_pni:
        FLAG_saved = False
        if str(query[0][1][0]) in symmetry_relations or str(query[0][1][1]) in symmetry_relations:
            symmetry_queries_pni.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in symmetry_relations and (not FLAG_saved):
            symmetry_queries_pni.add(query)
            FLAG_saved = True

    inverse_queries_pni = set()
    for query in queries_pni:
        FLAG_saved = False
        if str(query[0][1][0]) in inverse_relations or str(query[0][1][1]) in inverse_relations:
            inverse_queries_pni.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in inverse_relations and (not FLAG_saved):
            inverse_queries_pni.add(query)

    composition_queries_pni = set()
    for query in queries_pni:
        FLAG_saved = False
        if str(query[0][1][0]) in composition_relations or str(query[0][1][1]) in composition_relations:
            composition_queries_pni.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in composition_relations and (not FLAG_saved):
            composition_queries_pni.add(query)

    containment_queries_pni = set()
    for query in queries_pni:
        FLAG_saved = False
        if str(query[0][1][0]) in containment_relations or str(query[0][1][1]) in containment_relations:
            containment_queries_pni.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in containment_relations and (not FLAG_saved):
            containment_queries_pni.add(query)

    transtivity_queries_pni = set()
    for query in queries_pni:
        FLAG_saved = False
        if str(query[0][1][0]) in transtivity_relations or str(query[0][1][1]) in transtivity_relations:
            transtivity_queries_pni.add(query)
            FLAG_saved = True

        if str(query[1][1][0]) in transtivity_relations and (not FLAG_saved):
            transtivity_queries_pni.add(query)

    ## 2u (('e', ('r',)), ('e', ('r',)), ('u',))
    symmetry_queries_2u = set()
    for query in queries_2u:
        FLAG_saved = False
        for p in query[:2]:
            if str(p[1][0]) in symmetry_relations:
                symmetry_queries_2u.add(query)
                FLAG_saved =True
                break

    inverse_queries_2u = set()
    for query in queries_2u:
        for p in query[:2]:
            if str(p[1][0]) in inverse_relations:
                inverse_queries_2u.add(query)
                break

    composition_queries_2u = set()
    for query in queries_2u:
        for p in query[:2]:
            if str(p[1][0]) in composition_relations:
                composition_queries_2u.add(query)
                break

    containment_queries_2u = set()
    for query in queries_2u:
        for p in query[:2]:
            if str(p[1][0]) in containment_relations:
                containment_queries_2u.add(query)
                break

    transtivity_queries_2u = set()
    for query in queries_2u:
        for p in query[:2]:
            if str(p[1][0]) in transtivity_relations:
                transtivity_queries_2u.add(query)
                break

    ## up ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
    symmetry_queries_up = set()
    for query in queries_up:
        FLAG_saved = False
        if str(query[1][0]) in symmetry_relations:
            symmetry_queries_up.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0][:2]:
                if str(p[1][0]) in symmetry_relations:
                    symmetry_queries_up.add(query)
                    break

    inverse_queries_up = set()
    for query in queries_up:
        FLAG_saved = False
        if str(query[1][0]) in inverse_relations:
            inverse_queries_up.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0][:2]:
                if str(p[1][0]) in inverse_relations:
                    inverse_queries_up.add(query)
                    break

    composition_queries_up = set()
    for query in queries_up:
        FLAG_saved = False
        if str(query[1][0]) in composition_relations:
            composition_queries_up.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0][:2]:
                if str(p[1][0]) in composition_relations:
                    composition_queries_up.add(query)
                    break
    
    containment_queries_up = set()
    for query in queries_up:
        FLAG_saved = False
        if str(query[1][0]) in containment_relations:
            containment_queries_up.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0][:2]:
                if str(p[1][0]) in containment_relations:
                    containment_queries_up.add(query)
                    break
    
    transtivity_queries_up = set()
    for query in queries_up:
        FLAG_saved = False
        if str(query[1][0]) in transtivity_relations:
            transtivity_queries_up.add(query)
            FLAG_saved = True

        if not FLAG_saved:
            for p in query[0][:2]:
                if str(p[1][0]) in transtivity_relations:
                    transtivity_queries_up.add(query)
                    break

    test_symmetry_queries = {}
    test_symmetry_queries[('e', ('r',))] = symmetry_queries_1p
    test_symmetry_queries[('e', ('r', 'r'))] = symmetry_queries_2p
    test_symmetry_queries[('e', ('r', 'r', 'r'))] = symmetry_queries_3p
    test_symmetry_queries[(('e', ('r',)), ('e', ('r',)))] = symmetry_queries_2i
    test_symmetry_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = symmetry_queries_3i
    test_symmetry_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = symmetry_queries_ip
    test_symmetry_queries[(('e', ('r', 'r')), ('e', ('r',)))] = symmetry_queries_pi
    test_symmetry_queries[(('e', ('r',)), ('e', ('r', 'n')))] = symmetry_queries_2in
    test_symmetry_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = symmetry_queries_3in
    test_symmetry_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = symmetry_queries_inp
    test_symmetry_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = symmetry_queries_pin
    test_symmetry_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = symmetry_queries_pni
    test_symmetry_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = symmetry_queries_2u
    test_symmetry_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = symmetry_queries_up

    test_inverse_queries = {}
    test_inverse_queries[('e', ('r',))] = inverse_queries_1p
    test_inverse_queries[('e', ('r', 'r'))] = inverse_queries_2p
    test_inverse_queries[('e', ('r', 'r', 'r'))] = inverse_queries_3p
    test_inverse_queries[(('e', ('r',)), ('e', ('r',)))] = inverse_queries_2i
    test_inverse_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = inverse_queries_3i
    test_inverse_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = inverse_queries_ip
    test_inverse_queries[(('e', ('r', 'r')), ('e', ('r',)))] = inverse_queries_pi
    test_inverse_queries[(('e', ('r',)), ('e', ('r', 'n')))] = inverse_queries_2in
    test_inverse_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = inverse_queries_3in
    test_inverse_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = inverse_queries_inp
    test_inverse_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = inverse_queries_pin
    test_inverse_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = inverse_queries_pni
    test_inverse_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = inverse_queries_2u
    test_inverse_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = inverse_queries_up

    test_composition_queries = {}
    test_composition_queries[('e', ('r',))] = composition_queries_1p
    test_composition_queries[('e', ('r', 'r'))] = composition_queries_2p
    test_composition_queries[('e', ('r', 'r', 'r'))] = composition_queries_3p
    test_composition_queries[(('e', ('r',)), ('e', ('r',)))] = composition_queries_2i
    test_composition_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = composition_queries_3i
    test_composition_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = composition_queries_ip
    test_composition_queries[(('e', ('r', 'r')), ('e', ('r',)))] = composition_queries_pi
    test_composition_queries[(('e', ('r',)), ('e', ('r', 'n')))] = composition_queries_2in
    test_composition_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = composition_queries_3in
    test_composition_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = composition_queries_inp
    test_composition_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = composition_queries_pin
    test_composition_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = composition_queries_pni
    test_composition_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = composition_queries_2u
    test_composition_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = composition_queries_up

    test_containment_queries = {}
    test_containment_queries[('e', ('r',))] = containment_queries_1p
    test_containment_queries[('e', ('r', 'r'))] = containment_queries_2p
    test_containment_queries[('e', ('r', 'r', 'r'))] = containment_queries_3p
    test_containment_queries[(('e', ('r',)), ('e', ('r',)))] = containment_queries_2i
    test_containment_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = containment_queries_3i
    test_containment_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = containment_queries_ip
    test_containment_queries[(('e', ('r', 'r')), ('e', ('r',)))] = containment_queries_pi
    test_containment_queries[(('e', ('r',)), ('e', ('r', 'n')))] = containment_queries_2in
    test_containment_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = containment_queries_3in
    test_containment_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = containment_queries_inp
    test_containment_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = containment_queries_pin
    test_containment_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = containment_queries_pni
    test_containment_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = containment_queries_2u
    test_containment_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = containment_queries_up

    test_transtivity_queries = {}
    test_transtivity_queries[('e', ('r',))] = transtivity_queries_1p
    test_transtivity_queries[('e', ('r', 'r'))] = transtivity_queries_2p
    test_transtivity_queries[('e', ('r', 'r', 'r'))] = transtivity_queries_3p
    test_transtivity_queries[(('e', ('r',)), ('e', ('r',)))] = transtivity_queries_2i
    test_transtivity_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = transtivity_queries_3i
    test_transtivity_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = transtivity_queries_ip
    test_transtivity_queries[(('e', ('r', 'r')), ('e', ('r',)))] = transtivity_queries_pi
    test_transtivity_queries[(('e', ('r',)), ('e', ('r', 'n')))] = transtivity_queries_2in
    test_transtivity_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = transtivity_queries_3in
    test_transtivity_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = transtivity_queries_inp
    test_transtivity_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = transtivity_queries_pin
    test_transtivity_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = transtivity_queries_pni
    test_transtivity_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = transtivity_queries_2u
    test_transtivity_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = transtivity_queries_up

    test_joint_queries = {}
    test_joint_queries[('e', ('r',))] = symmetry_queries_1p & composition_queries_1p # & inverse_queries_1p 
    test_joint_queries[('e', ('r', 'r'))] = symmetry_queries_2p & composition_queries_2p #& inverse_queries_2p 
    test_joint_queries[('e', ('r', 'r', 'r'))] = symmetry_queries_3p & composition_queries_3p #& inverse_queries_3p
    test_joint_queries[(('e', ('r',)), ('e', ('r',)))] = symmetry_queries_2i & composition_queries_2i # & inverse_queries_2i
    test_joint_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = symmetry_queries_3i & composition_queries_3i #& inverse_queries_3i
    test_joint_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = symmetry_queries_ip & composition_queries_ip #& inverse_queries_ip
    test_joint_queries[(('e', ('r', 'r')), ('e', ('r',)))] = symmetry_queries_pi & composition_queries_pi #& inverse_queries_pi
    test_joint_queries[(('e', ('r',)), ('e', ('r', 'n')))] = symmetry_queries_2in & composition_queries_2in #& inverse_queries_2in
    test_joint_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = symmetry_queries_3in & composition_queries_3in #& inverse_queries_3in
    test_joint_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = symmetry_queries_inp & composition_queries_inp #& inverse_queries_inp
    test_joint_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = symmetry_queries_pin & composition_queries_pin #& inverse_queries_pin
    test_joint_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = symmetry_queries_pni & composition_queries_pni #& inverse_queries_pni
    test_joint_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = symmetry_queries_2u & composition_queries_2u #& inverse_queries_2u
    test_joint_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = symmetry_queries_up & composition_queries_up #& inverse_queries_up 

    test_other_queries = {}
    test_other_queries[('e', ('r',))] = queries_1p - symmetry_queries_1p - inverse_queries_1p - composition_queries_1p - containment_queries_1p - transtivity_queries_1p
    test_other_queries[('e', ('r', 'r'))] = queries_2p - symmetry_queries_2p -inverse_queries_2p - composition_queries_2p - containment_queries_2p - transtivity_queries_2p
    test_other_queries[('e', ('r', 'r', 'r'))] = queries_3p - symmetry_queries_3p - inverse_queries_3p - composition_queries_3p - containment_queries_3p - transtivity_queries_3p 
    test_other_queries[(('e', ('r',)), ('e', ('r',)))] = queries_2i - symmetry_queries_2i - inverse_queries_2i - composition_queries_2i - containment_queries_2i - transtivity_queries_2i
    test_other_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r',)))] = queries_3i - symmetry_queries_3i - inverse_queries_3i - composition_queries_3i - containment_queries_3i - transtivity_queries_3i
    test_other_queries[((('e', ('r',)), ('e', ('r',))), ('r',))] = queries_ip - symmetry_queries_ip - inverse_queries_ip - composition_queries_ip - containment_queries_ip - transtivity_queries_ip
    test_other_queries[(('e', ('r', 'r')), ('e', ('r',)))] = queries_pi - symmetry_queries_pi - inverse_queries_pi - composition_queries_pi - containment_queries_pi - transtivity_queries_pi
    test_other_queries[(('e', ('r',)), ('e', ('r', 'n')))] = queries_2in - symmetry_queries_2in - inverse_queries_2in - composition_queries_2in - containment_queries_2in - transtivity_queries_2in
    test_other_queries[(('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n')))] = queries_3in - symmetry_queries_3in - inverse_queries_3in - composition_queries_3in - containment_queries_3in - transtivity_queries_3in
    test_other_queries[((('e', ('r',)), ('e', ('r', 'n'))), ('r',))] = queries_inp - symmetry_queries_inp - inverse_queries_inp - composition_queries_inp - containment_queries_inp - transtivity_queries_inp
    test_other_queries[(('e', ('r', 'r')), ('e', ('r', 'n')))] = queries_pin - symmetry_queries_pin - inverse_queries_pin - composition_queries_pin - containment_queries_pin - transtivity_queries_pin
    test_other_queries[(('e', ('r', 'r', 'n')), ('e', ('r',)))] = queries_pni - symmetry_queries_pni - inverse_queries_pni - composition_queries_pni - containment_queries_pni - transtivity_queries_pni
    test_other_queries[(('e', ('r',)), ('e', ('r',)), ('u',))] = queries_2u - symmetry_queries_2u - inverse_queries_2u - composition_queries_2u - containment_queries_2u - transtivity_queries_2u
    test_other_queries[((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))] = queries_up - symmetry_queries_up - inverse_queries_up - composition_queries_up - containment_queries_up - transtivity_queries_up
    
    return test_symmetry_queries, test_inverse_queries, test_composition_queries, test_containment_queries, test_transtivity_queries, test_joint_queries, test_other_queries
