# Datos de entrada (`data/raw/`)

Coloca aquí los tres archivos CSV originales del trabajo parcial cuando los tengas en disco:

- `feedback_curso_universitario.csv`
- `feedback_curso_universitario2.csv`
- `feedback_curso_universitario3.csv`

## Comportamiento automático de `src/dataset_builder.py`

1. **Modo real (`real_csv`)**: si los archivos CSV anteriores están presentes en `data/raw/` (o en la raíz del proyecto), el pipeline los carga, unifica y aplica exactamente las reglas de saneamiento documentadas en `Informe_Parcial_SisInt.pdf` y `Parcial_SI_corregido.ipynb`:
   - Eliminación de cabeceras repetidas (`id == 'id'`).
   - Normalización de la columna `curso` (`Sistemas opertativos` → `Sistemas operativos`).
   - Canonización de las 9 variantes de `polaridad_basica` a `{positivo, negativo, neutral}`.
   - Parseo estructurado de `aspectos` (`json.loads` / `ast.literal_eval`).
   - Aplicación de la guía de desempate LLM por llave compuesta `(origen_archivo, id)` en los casos auditados (IDs 3, 6, 9 y 11 del archivo 1).
2. **Modo demostrativo fiel al parcial (`synthetic_parcial_replica`)**: si los CSV originales no se encuentran en la carpeta, el generador construye automáticamente un dataset reproducible de 618 registros en español con la misma distribución de clases (41.7% neutral, 33.2% negativo, 25.1% positivo), las mismas 7 asignaturas, los mismos 4 semestres, ~76% de comentarios con cláusulas contrastantes, los casos auditados del informe y anotaciones estructuradas de los 9 aspectos docentes. Esto permite ejecutar y validar todo el flujo end-to-end de inmediato.
