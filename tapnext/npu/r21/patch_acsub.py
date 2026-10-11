# round 21 ACSUB: the attention's O panels release C sub-parts per tile (tile t's panels = bytes [3,456 t, 3,456 (t + 1)) = sub-part t)
p = "mm8.cc"
s = open(p).read()
def rep(old, new):
    global s
    assert s.count(old) == 1, old[:70]
    s = s.replace(old, new)
rep('''        ps.push(pr.template to_vector<bfp16ebs8>());
      }
    }
  }
#ifdef ATTK_COUNT''', '''        ps.push(pr.template to_vector<bfp16ebs8>());
      }
    }
#ifdef CSUB  // round 21 ACSUB: tile t's panels are C bytes [3,456 t, 3,456 (t + 1)) = sub-part t: let it drain now
    release(c_out, 48 + CS_CONS, 1);
#endif
  }
#ifdef ATTK_COUNT''')
rep('''#ifdef CSUB
  release(c_out, 48 + CS_CONS, 3);  // round 21: part 1's three sub-parts (IRON's out_c.release is part 2)''', '''#ifdef CSUB
  release(c_out, 48 + CS_CONS, 3 - g_ntiles);  // round 21: the sub-parts no tile released (IRON's out_c.release is part 2)''')
open(p, "w").write(s)
print("ACSUB applied")
