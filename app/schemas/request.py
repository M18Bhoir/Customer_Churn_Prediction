from pydantic import BaseModel, Field, validator


class CustomerData(BaseModel):
    """Schema for customer data input"""
    customer_id: int = Field(..., description="Unique customer identifier")
    age: float = Field(..., ge=18, le=100, description="Customer age")
    tenure_months: float = Field(..., ge=0, description="Months as customer")
    monthly_charges: float = Field(..., ge=0, description="Monthly charges in currency")
    total_charges: float = Field(..., ge=0, description="Total charges to date")
    contract_type: str = Field(..., description="Contract type")
    internet_service: str = Field(..., description="Internet service type")
    support_calls: float = Field(..., ge=0, description="Number of support calls")
    late_payments: int = Field(..., ge=0, description="Number of late payments")
    
    @validator('contract_type')
    def validate_contract_type(cls, v):
        valid_types = ['Month-to-month', 'One year', 'Two year']
        if v not in valid_types:
            raise ValueError(f"contract_type must be one of {valid_types}")
        return v
    
    @validator('internet_service')
    def validate_internet_service(cls, v):
        valid_services = ['DSL', 'Fiber optic', 'No']
        if v not in valid_services:
            raise ValueError(f"internet_service must be one of {valid_services}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": 12345,
                "age": 35.0,
                "tenure_months": 24.0,
                "monthly_charges": 79.99,
                "total_charges": 1919.76,
                "contract_type": "One year",
                "internet_service": "Fiber optic",
                "support_calls": 2.0,
                "late_payments": 0
            }
        }