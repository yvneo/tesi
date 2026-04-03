import statistical_analysis as sa
def main():
    df = sa.load_data()
    sa.perform_statistical_tests(df)

if __name__ == "__main__":
    main()