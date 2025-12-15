import asyncio
from app.db import Session, Order, init_db

async def seed():
    await init_db()
    async with Session() as s:
        s.add_all([
            Order(order_no="A12345", customer="Acme Schools", status="Shipped", notes="Left warehouse 2025-11-02"),
            Order(order_no="B77777", customer="Springfield HS", status="Pending Payment", notes="PO awaiting approval")
        ])
        await s.commit()

asyncio.run(seed())