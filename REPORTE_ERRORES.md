# Reporte de Errores e Inconsistencias - GlauberPT.jl

**Fecha:** 10 de diciembre de 2025  
**Análisis de:** `/src` folder completo

---

## 🔴 ERRORES CRÍTICOS

### 1. Variable `bond` no definida en `build_sequential_transition_tt`
**Archivo:** `tensor_train_efficient/transition_rates_builder.jl` (línea ~245)  
**Problema:**
```julia
compress!(tensors, svd_trunc=TruncBond(bond))
```
La variable `bond` no está definida. No es un parámetro de la función ni está en scope.

**Solución:**
Agregar `bond` como parámetro de la función:
```julia
function build_sequential_transition_tt(transition_rate, params, bond::Int, Q::Int = 2, σ = x -> 2x - 3)
```

---

### 2. Variable `N` no definida en `parallel_transition_tensor_train`
**Archivo:** `tensor_train_efficient/transition_rates_builder.jl` (líneas ~286-287)  
**Problema:**
```julia
params_1 = (N = N, beta = params.betas[1], ...)
params_2 = (N = N, beta = params.betas[2], ...)
```
La variable `N` no existe. Debería ser `params.N`.

**Solución:**
```julia
params_1 = (N = params.N, beta = params.betas[1], ...)
params_2 = (N = params.N, beta = params.betas[2], ...)
```

---

### 3. Llamada incorrecta a `build_transition_tensortrain`
**Archivo:** `tensor_train_efficient/transition_rates_builder.jl` (línea ~288)  
**Problema:**
```julia
A1 = build_transition_tensortrain(params_1; update_rule, Q, σ)
```
- `params_1` es un NamedTuple, pero la función espera un `MCParameters`
- Falta pasar el parámetro `bond` si se corrige el error #1

**Solución:**
Cambiar la firma de `build_transition_tensortrain` o convertir los NamedTuples en `MCParameters`.

---

### 4. Inconsistencia en la firma de `accumulate_observables!`
**Archivo:** `monte_carlo_efficient/observables.jl` vs `monte_carlo_efficient/monte_carlo_general.jl`  
**Problema:**
- La función se define con 8 parámetros en `observables.jl` (línea ~102)
- Se llama con 9 parámetros en `monte_carlo_general.jl` (líneas ~139, 202)

**Llamada actual:**
```julia
accumulate_observables!(obs_accumulators, chains, chain_energies, 1, sample, observables, 
                       params.j_vector, params.h_vector, params.betas)
```

**Definición:**
```julia
function accumulate_observables!(accumulators, chains, energies, t, sample, observables,
                                j_vector, h_vector, betas)
```

**Solución:**
La definición parece correcta con 9 parámetros. Verificar que todas las llamadas usen este formato.

---

### 5. Cálculo de energía duplicado e ineficiente
**Archivo:** `monte_carlo_efficient/observables.jl` (línea ~114)  
**Problema:**
```julia
elseif obs == :energy
    for c in 1:n_chains
        E = compute_total_energy(chains[c], j_vector, h_vector)
        # E = energies[c]  # <-- COMENTADO
```
Se recalcula la energía cuando ya está disponible en `energies[c]`.

**Solución:**
Descomentar la línea correcta:
```julia
E = energies[c]
```

---

### 6. Función `compute_total_energy` duplicada
**Archivo:** `monte_carlo_efficient/observables.jl` y `monte_carlo_efficient/swap_criteria.jl`  
**Problema:**
La función `compute_total_energy` está definida en dos lugares, causando redundancia.

**Solución:**
Eliminar una de las definiciones y asegurarse de que ambos módulos importen correctamente.

---

## ⚠️ ERRORES MODERADOS

### 7. Uso de `fixed_rate_swap` sin actualizar energías
**Archivo:** `monte_carlo_efficient/monte_carlo_general.jl` (línea ~183)  
**Problema:**
```julia
if swap_criterion == :fixed_rate
    apply_fixed_rate_swap!(chains, params.s, rng)
    # AGREGAR: Después del swap, actualizar el orden de energías también
```
Después del swap, las energías en `chain_energies` no se reordenan, causando inconsistencia.

**Solución:**
Implementar una versión de `fixed_rate_swap` que también intercambie energías, o recalcularlas después del swap.apply_fixed_rate_swap

---

### 8. Parámetros comentados incorrectamente en `parameters.jl`
**Archivo:** `monte_carlo_efficient/parameters.jl` (líneas 1-150)  
**Problema:**
Gran parte del código está comentado, incluyendo definiciones importantes y documentación.

**Solución:**
Eliminar el código comentado o descomentar lo necesario.

---

### 9. Orden de includes incorrecto en `GlauberPT.jl`
**Archivo:** `GlauberPT.jl` (líneas 53-65)  
**Problema potencial:**
Si los archivos tienen dependencias entre sí, el orden actual puede causar errores de "undefined".

**Recomendación:**
Verificar que:
1. `parameters.jl` se incluya primero (define `MCParameters`)
2. Los otros archivos sigan después

---

### 10. Exportaciones de funciones no existentes
**Archivo:** `GlauberPT.jl` (líneas 11-47)  
**Problema:**
Se exportan muchas funciones, pero algunas podrían no estar definidas o tener nombres incorrectos:
- `mult_sep` (debería ser `mult_sep_transition`?)
- `k_step_transition_tt` 
- Etc.

**Solución:**
Verificar que todas las funciones exportadas existan con esos nombres exactos.

---

## 📝 ADVERTENCIAS E INCONSISTENCIAS

### 11. Nomenclatura inconsistente
- `transition_rate_inertia` vs `glauber_transition_rate`
- `compute_total_energy` definido dos veces
- `mult_sep` vs `mult_sep_transition`

### 12. Falta de validación de índices
En `transition_rate_inertia` (línea ~365):
```julia
params.p0* (sigma_new == sigma_neighbors[site_index == 1 ? 1 : site_index == N ? 2 : 2] ? 1.0 : 0.0)
```
Esta expresión es confusa y propensa a errores.

### 13. Uso de arrays multidimensionales con `;;;` y `;;;;`
**Archivos:** `build_sequential_transition_tt`  
**Problema:**
Sintaxis como `[1 ;;; 0 ;;;; 0 ;;; 1]` requiere Julia ≥ 1.7 y puede ser confusa.

**Recomendación:**
Usar `reshape` o `cat` para mayor claridad:
```julia
reshape([1, 0, 0, 1], 1, 1, 2, 2)
```

### 14. Comentarios obsoletos
Varios comentarios en español mezclados con código, algunos obsoletos o incorrectos.

### 15. Falta de manejo de errores
No hay validación de que:
- `betas` tenga al menos 2 elementos cuando se usa swap
- `j_vector` y `h_vector` tengan las longitudes correctas
- Los parámetros de funciones sean del tipo correcto

---

## 🔧 RECOMENDACIONES DE REFACTORIZACIÓN

### 1. Estructura de parámetros
Considerar usar un solo tipo de parámetros (`MCParameters`) en todos lados, en vez de mezclar `MCParameters` y `NamedTuple`.

### 2. Manejo de energías
Centralizar el cálculo y actualización de energías para evitar inconsistencias.

### 3. Documentación
- Completar docstrings faltantes
- Agregar ejemplos de uso
- Documentar el formato esperado de arrays multidimensionales

### 4. Tests
Agregar tests unitarios para:
- Construcción de tensor trains
- Cálculo de observables
- Swaps con/sin energías
- Validación de parámetros

### 5. Separación de concerns
Considerar separar:
- Funciones matemáticas puras (cálculos de energía, transiciones)
- Manejo de estado (cadenas, energías)
- I/O y resultados

---

## ✅ PRIORIDADES DE CORRECCIÓN

**Alta prioridad (errores que impiden ejecución):**
1. Variable `bond` no definida (#1)
2. Variable `N` no definida (#2)
3. Firma de `accumulate_observables!` (#4)

**Media prioridad (errores que causan resultados incorrectos):**
5. Cálculo de energía duplicado (#5)
6. Swap sin actualizar energías (#7)
7. Llamada incorrecta a `build_transition_tensortrain` (#3)

**Baja prioridad (mejoras de código):**
8-15. Resto de advertencias y recomendaciones

---

## 📊 RESUMEN

- **Errores críticos:** 6
- **Errores moderados:** 4
- **Advertencias:** 5
- **Recomendaciones:** 5

**Total de issues:** 20

---

**Nota:** Este reporte se generó mediante análisis estático del código. Se recomienda ejecutar tests para verificar el comportamiento real y detectar errores en runtime.
