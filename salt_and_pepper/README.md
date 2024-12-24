Задание: 
 Given the image of size M×N with “Salt and Pepper” noise, implement and apply a CUDA version of 9-point
 median filter and store the result to output image. Missing values for edge rows and columns are to be taken from
 nearest pixels. CUDA implementation must make use of texture memory.
Реализация:
1. Добавление шума: Функция add_salt_and_pepper_noise добавляет шум в изображение.
2. Фильтрация на CPU: Функция apply_median_filter_cpu выполняет медианную фильтрацию изображения на центральном процессоре.
3. Фильтрация на GPU: Функция apply_median_filter_gpu выполняет аналогичную операцию, но на графическом процессоре (GPU).
4. Подготовка и выполнение фильтра на GPU: Функция prepare_and_execute_gpu подготавливает данные для работы на GPU, распределяет вычисления по блокам и потокам, а затем вызывает apply_median_filter_gpu для выполнения медианной фильтрации на GPU.
5. Тестирование: В функции test изображение загружается, на него накладывается шум, затем применяются медианные фильтры на CPU и GPU. Результаты отображаются в виде четырёх изображений: исходное, зашумленное, отфильтрованное на CPU и отфильтрованное на GPU. Также сохраняются изображения в файлы и выводятся время выполнения на CPU и GPU, а также ускорение.

