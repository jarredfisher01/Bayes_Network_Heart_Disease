import model
import pyAgrum as gum
def main():

    bn = model.build_model()
    
    model.inference(bn)

    dn = model.build_model_decision()
    
    model.inference_decision(dn)
   
    

if __name__ == "__main__":
    main()