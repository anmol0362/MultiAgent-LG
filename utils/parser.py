def parse_steps(plan_text: str) -> list[str]:
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line[0].isdigit() and "." in line:
            steps.append(line.split(".", 1)[1].strip())
        elif line.lower().startswith("step"):
            parts = line.split(":", 1)
            steps.append(parts[-1].strip())

    return steps[:5]