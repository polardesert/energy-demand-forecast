#!/usr/bin/env python
"""
Plot functions
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as m_dates

__author__ = "Usman Ahmad"
__version__ = "1.0.1"


def line_plot_date(
        idf: pd.DataFrame,
        x_ax: str,
        y_ax: list,
        title: str,
        labels: tuple,
        output: str,
        markers=None,  # dict of tuples: [ (col_name, marker), ... ]
        date_grid_major="y",  # y = years, m = months, d = days
        date_grid_minor="m",  # y = years, m = months, d = days
        date_format="%Y-%m",
        fig_size=(10, 5),
        legend=True,
        vertical_marker=False
) -> None:
    df = idf.copy()
    df.reset_index(inplace=True)
    df[x_ax] = pd.to_datetime(df[x_ax])
    years = m_dates.YearLocator()  # every year
    months = m_dates.MonthLocator()  # every month
    days = m_dates.DayLocator()  # every month
    hours = m_dates.HourLocator()  # every hour
    minutes = m_dates.MinuteLocator()  # every minute
    years_fmt = m_dates.DateFormatter(date_format)

    fig, ax = plt.subplots(figsize=fig_size)
    for _y_ax in y_ax:
        ax.plot(x_ax, _y_ax, data=df)

    # plot markers
    markers = {} if markers is None else markers
    for marker_name, marker_info in markers.items():
        # get index values where marker is True
        marker_array = list(df[df[marker_name]].index.values)
        marker_target, marker_shape = marker_info
        marker_color = marker_shape[0]

        # plot markers
        ax.plot(
            df[x_ax],
            df[marker_target],
            marker_shape,
            markersize=12,
            markevery=marker_array
        )
        if vertical_marker:
            for marker_val in marker_array:
                ax.axvline(x=df[x_ax][marker_val], color=marker_color)

    # format the ticks
    loc_map = {"y": years, "m": months, "d": days, "h": hours, "i": minutes}
    ax.xaxis.set_major_locator(loc_map[date_grid_major])
    ax.xaxis.set_major_formatter(years_fmt)
    ax.xaxis.set_minor_locator(loc_map[date_grid_minor])

    # noinspection SpellCheckingInspection
    datemin, datemax = min(df[x_ax]), max(df[x_ax])
    ax.set_xlim(datemin, datemax)

    ax.set(xlabel=labels[0], ylabel=labels[1], title=title)
    ax.grid(True)
    ax.axis('tight')
    if legend:
        ax.legend(
            loc='best', ncol=1, bbox_to_anchor=(1, 0.5)
        )

    fig.autofmt_xdate()
    if output is not None:
        plt.savefig(output, bbox_inches='tight')


def line_plot(
        df: pd.DataFrame,
        x_ax: str,
        y_ax: list,
        title: str,
        labels: tuple,
        fig_size=(10, 5)
) -> None:
    fig, ax = plt.subplots(figsize=fig_size)
    for _y_ax in y_ax:
        ax.plot(x_ax, _y_ax, data=df)

    ax.set(xlabel=labels[0], ylabel=labels[1], title=title)
    ax.grid(True)
    ax.axis('tight')

    plt.show()
