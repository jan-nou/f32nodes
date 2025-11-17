from f32nodes import Runner


def main():
    runner = Runner("graph.yaml")
    runner.fps = 20  # adjust to your system performance
    return runner.start_app()


if __name__ == "__main__":
    main()
