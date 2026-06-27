from pathlib import Path

import numpy as np
from pydantic import BaseModel

DATA_PATH = Path(__file__).parent / "data"
DEBUG_PATH = Path(__file__).parent / ".debug"

SERVICE_IDS = [
    564809,
    564375,
    695686,
    564795,
    564969,
    693243,
    693700,
]


class DemandMatrixCell(BaseModel):
    csName: str
    csOrder: int
    price: float
    dayX: int
    demand: float


class DemandMatrix(BaseModel):
    values: list[DemandMatrixCell]


def get_env_data(service_ids: list[int]) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], list[str], list[int]]:
    """Utils function to load the environment data."""

    matrices_and_prices: list[tuple[int, np.ndarray, np.ndarray]] = []
    bucket_names: list[str] = []
    day_xs: list[int] = []

    for service_id in service_ids:
        with open(DATA_PATH / f"demand_matrix_{service_id}.json") as f:
            demand_matrix = DemandMatrix.model_validate_json(f.read())

        demand_by_day_x = {}
        for cell in demand_matrix.values:
            demand_by_day_x.setdefault(cell.dayX, []).append(
                {
                    "demand": cell.demand,
                    "price": cell.price,
                    "csName": cell.csName,
                    "csOrder": cell.csOrder,
                    "dayX": cell.dayX,
                }
            )
        n_day_xs = len(demand_by_day_x)
        n_buckets = len(demand_by_day_x[-2])  # assuming all day_x have same number of buckets

        demand_matrix = np.zeros((n_buckets, n_day_xs), dtype=int)

        prices = np.array([d["price"] for d in sorted(demand_by_day_x[-2], key=lambda d: d["csOrder"], reverse=True)])

        if not bucket_names:
            bucket_names = [d["csName"] for d in sorted(demand_by_day_x[-2], key=lambda d: d["csOrder"], reverse=True)]
        else:
            if bucket_names != [
                d["csName"] for d in sorted(demand_by_day_x[-2], key=lambda d: d["csOrder"], reverse=True)
            ]:
                raise ValueError("Bucket names do not match across services")

        day_xs = []
        for i, (day_x, demands) in enumerate(sorted(demand_by_day_x.items(), key=lambda item: item[0])):
            day_xs.append(day_x)
            demand_matrix[:, i] = [
                round(demand["demand"]) for demand in sorted(demands, key=lambda d: d["csOrder"], reverse=True)
            ]

        # weird demand on D0 because of early flights
        # demand_matrix[:, -1] = np.max(demand_matrix[:, -1], np.ones(n_buckets))
        demand_matrix[:, -1] = demand_matrix[:, -2]

        matrices_and_prices.append((service_id, demand_matrix, prices))

    return matrices_and_prices, bucket_names, day_xs
