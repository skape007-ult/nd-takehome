from nd_verify import verify_text

proof = """THM ( P v Q ) , ( ~ P ) , ( P > Q ) SEQ Q PRF
N1 ( P v Q ) : PR ;
N2 ( ~ P ) : PR ;
N3 ( P > Q ) : PR ;
N4 | P : AS ;
N5 | F : NEGE N4 N2 ;
N6 | Q : BOTE N5 ;
N7 | Q : AS ;
N8 Q : ORE N1 N4 N6 N7 N7 ;
QED"""

print(verify_text(proof))