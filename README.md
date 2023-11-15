CUDA Radix Sort Program
Overview

This program implements a radix sort algorithm using CUDA, making it suitable for sorting large arrays of unsigned integers on NVIDIA GPUs. Radix sort is a non-comparative integer sorting algorithm that sorts data with integer keys by grouping keys by individual digits that share the same significant position and value.
Features

    Efficient Sorting: Utilizes the parallel processing power of GPUs for sorting large datasets.
    User Input: Allows users to specify the size of the array and input each element manually.
    Error Checking: Includes basic error checking for CUDA operations (omitted in the code snippet but recommended for full implementation).

Requirements

    NVIDIA GPU with CUDA Compute Capability 3.5 or higher.
    CUDA Toolkit (latest version recommended).

Installation

    Ensure that the CUDA Toolkit is installed on your system.
    Clone this repository or download the source code.

Code Structure

    plus_scan: A device function for performing an exclusive scan, used in the sorting process.
    partition_by_bit: A device function for partitioning the array based on a specific bit.
    radix_sort: The main kernel function that orchestrates the radix sort process across multiple bits.
    main: The host function that handles user input, memory allocation, and kernel invocation.

Limitations

    Currently handles only arrays of unsigned int.
    Requires manual input for array size and elements, which might not be practical for very large arrays.
    Error handling in the code is minimal and should be expanded for production use.
