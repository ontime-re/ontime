<div align="center">
    <img src="res/ontime-logo.png" width="180" title="hover text">
    <h1>onTime—<i>your library to work with time series</i></h1>
</div>

[![Continuous Integration](https://github.com/fredmontet/ontime/actions/workflows/ci.yml/badge.svg)](https://github.com/fredmontet/ontime/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/ontime.svg)](https://badge.fury.io/py/ontime)

## Getting Started

For now, the best is to look at the examples in the `notebooks` folder.

## Purpose of the library

The purpose of onTime is to make a technological transfer to the partners of the 
DiagnoBat project with respect to their intellectual property. Most parts of 
the library are extendable by using dynamically loaded classes. This
mecanism allows anyone to keep parts of the library private within their own company. 

The library objectives are :

  1. to extend the time series libraries (Darts, GluonTS, Kats, etc.)
  2. to provide benchmarking tools for models and/or detectors
  3. to provide domain specific tools (e.g. for energy consumption, district heating networks, etc.)

In case you have any questions, don't hesitate to ask them by opening an issue or email.

## Contribute

We welcome contributions to onTime. If you want to contribute code, please open an issue first or assign 
yourself to one.

As of the 15th of January 2024, since the project is small, we add ourselves to the repository as
contributors and we don't use pull requests. We will use pull requests when the project will be bigger.

Still, please respect the following branch naming convention : 

    <issue number>-<issue slug>

For example, if you are working on issue #1, you should create a branch named `1-add-readme`. If you are
on an issue page, there is a button to create a branch with the correct name in the sidebar.

## Authors

Frédéric Montet (frederic.montet@hefr.ch)

## License

Apache License 2.0
