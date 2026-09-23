%%{init: {'theme': 'neutral'}}%%
sequenceDiagram
    autonumber
    actor Usuario
    participant Web
    participant API
    participant Modelo

    Usuario->>Web: Ingresa texto
        Usuario->>Web: (Opcional) Ingresa párrafo citado
    Usuario->>Web: Selecciona modelo de clasificación
    Usuario->>Web: Botón "Analizar Cita"
    Web->>API: Consulta API con modelo seleccionado
    API->>Modelo: Procesa solicitud
    Modelo-->>API: Retorna respuesta con porcentaje de clasificación
    API-->>Web: Retorna descripción de categoría
    Web-->>Usuario: Muestra resultados 
