import model

def main():
    bn = model.build_model()
    model.inference(bn, evidence={"Alcohol_Abuse":1})

if __name__ == "__main__":
    main()