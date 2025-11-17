from f32nodes import Runner


def main():
    runner = Runner("graph.yaml")
    return runner.start_app()


if __name__ == "__main__":
    main()
