# PHD Stefan paving

This repository contains a CUDA solver for a three-dimensional Stefan problem with multiple briquettes arranged inside a larger volume.

The model evolves a temperature field on a regular 3D grid, with separate initial temperatures for the briquettes and surrounding medium and temperature-dependent material properties around the phase transition. CUDA kernels perform the finite-difference update on the GPU.

The program is research-oriented code with experiment parameters embedded directly in the source. It writes a text summary and VTK snapshots that can be inspected later with scientific-visualization tools.

## Описание

Этот репозиторий содержит CUDA решатель трехмерной задачи Стефана с несколькими брикетами, расположенными внутри большего объема.

Модель рассчитывает изменение температурного поля на регулярной трехмерной сетке с разными начальными температурами брикетов и окружающей среды и с зависящими от температуры свойствами материала в области фазового перехода. Ядра CUDA выполняют конечно-разностное обновление на GPU.

Программа является исследовательским кодом с параметрами эксперимента, заданными непосредственно в исходниках. Она записывает текстовую сводку и снимки VTK, которые затем можно просматривать средствами научной визуализации.

## Сборка

Нужны CMake, CUDA Toolkit и совместимый NVIDIA GPU.

```sh
cmake --preset release
cmake --build --preset release
```

Исполняемый файл:

```text
build/release/paving
```

## Запуск

```sh
mkdir -p plot
./build/release/paving
```

Параметров командной строки нет. Температуры, размер брикетов, количество элементов, шаг сетки и шаг времени задаются непосредственно в `src/paving.cu`.

## Результат

Основные численные результаты записываются в `out.txt`. Поля температуры периодически сохраняются в бинарных VTK-файлах:

```text
plot/result_<time>.vtk
```

## Модель

Расчет использует трехмерную явную конечно-разностную схему. В начальном условии внутри объема размещается набор брикетов, окруженных другой средой; теплообмен и фазовый переход рассчитываются во времени на GPU.
