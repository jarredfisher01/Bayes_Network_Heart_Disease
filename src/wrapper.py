import pyAgrum as gum
import pyAgrum.lib.image as gumimage  

bn=gum.BayesNet('WaterSprinkler')
print(bn)

c=bn.add(gum.LabelizedVariable('c','cloudy ?',2))
print(c)

s, r, w = [ bn.add(name, 2) for name in "srw" ] #bn.add(name, 2) === bn.add(gum.LabelizedVariable(name, name, 2))
print (s,r,w)
print (bn)

for link in [(c,s),(c,r),(s,w),(r,w)]:
    bn.addArc(*link)
print(bn)

bn.cpt("c").fillWith([0.5,0.5])

bn.cpt("s")[0,:]=0.5 # equivalent to [0.5,0.5]
bn.cpt("s")[1,:]=[0.9,0.1]

bn.cpt("w")[{'r': 0, 's': 0}] = [1, 0]
bn.cpt("w")[{'r': 0, 's': 1}] = [0.1, 0.9]
bn.cpt("w")[{'r': 1, 's': 0}] = [0.1, 0.9]
bn.cpt("w")[{'r': 1, 's': 1}] = [0.01, 0.99]


bn.cpt("r")[{'c':0}]=[0.8,0.2]
bn.cpt("r")[{'c':1}]=[0.2,0.8]

print(gum.availableBNExts())

gum.saveBN(bn,"WaterSprinkler.bif")
with open("WaterSprinkler.bif","r") as out:
    print(out.read())

gumimage.export(bn,"test_export.png")

