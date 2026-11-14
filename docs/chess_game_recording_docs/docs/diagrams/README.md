# PlantUML diagrams

All files are PlantUML source (`.puml`) and use the corresponding UML diagram syntax. The styles are monochrome:

```plantuml
skinparam monochrome true
skinparam shadowing false
```

Render locally (requires Java + PlantUML jar):

```bash
java -jar plantuml.jar -tpng *.puml
java -jar plantuml.jar -tsvg *.puml
```

Or paste the content of a `.puml` file into the official online renderer: https://www.plantuml.com/plantuml/uml/ .

Files:

- `01_use_case.puml` — Use Case diagram.
- `02_component_architecture.puml` — Component diagram.
- `03_sequence_register_move.puml` — Sequence diagram.
- `04_state_machine.puml` — State Machine diagram.
- `05_activity_pipeline.puml` — Activity diagram.
- `06_class_model.puml` — Class diagram.

The `.puml` files are canonical sources. Images should be generated from these sources; do not manually redraw them as generic flowcharts.
