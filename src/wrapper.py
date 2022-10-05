import model

def main():
    bn = model.build_model_decision()
    
    model.inference_decision(bn,{'Heart_Disease':0})

if __name__ == "__main__":
    main()