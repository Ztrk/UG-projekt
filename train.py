from models import LPG

def main():
    lpg = LPG()
    lpg.train(10, 10000)

if __name__ == '__main__':
    main()
