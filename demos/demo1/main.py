from f32nodes import Runner


def main():
    # Runner resolves the graph path relative to this file before loading the YAML graph.
    runner = Runner("graph.yaml")
    runner.fps = 24  # default is 30 FPS; lowering leaves more time per frame
    runner.stats_print_interval = 500  # default is 1000 frames between reports
    print(
        f"[demo1] Runner configured with fps={runner.fps}, "
        f"stats_print_interval={runner.stats_print_interval} frames"
    )
    return runner.start_app()


if __name__ == "__main__":
    main()
