import model

def main():
    bn = model.build_model_decision()
    
    model.inference_decision(bn)

    # bn = model.build_model()
    
    # model.inference(bn)

if __name__ == "__main__":
    main()