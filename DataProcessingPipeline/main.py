# Data Processing Pipeline & API Layer

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import pandas as pd
import numpy as np
import PyPDF2
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import redis
from dataclasses import dataclass
from enum import Enum

# Database Models
Base = declarative_base()

class TransactionCategory(Enum):
    INCOME = "income"
    EXPENSE = "expense"
    TRANSFER = "transfer"

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    date = Column(DateTime)
    amount = Column(Float)
    description = Column(Text)
    category = Column(String)
    subcategory = Column(String)
    transaction_type = Column(String)
    balance_after = Column(Float)

class BusinessMetric(Base):
    __tablename__ = "business_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    metric_date = Column(DateTime)
    revenue = Column(Float)
    expenses = Column(Float)
    inventory_value = Column(Float)
    accounts_receivable = Column(Float)
    accounts_payable = Column(Float)

# Data Processing Classes
@dataclass
class ProcessedTransaction:
    date: datetime
    amount: float
    description: str
    category: str
    subcategory: str
    transaction_type: TransactionCategory
    balance_after: float

class DocumentProcessor:
    def __init__(self):
        self.expense_keywords = {
            'rent': ['rent', 'lease', 'property'],
            'utilities': ['electric', 'water', 'gas', 'internet', 'phone'],
            'supplies': ['supplies', 'materials', 'inventory', 'stock'],
            'marketing': ['advertising', 'marketing', 'promotion'],
            'transport': ['fuel', 'transport', 'delivery', 'shipping'],
            'professional': ['legal', 'accounting', 'consulting', 'professional']
        }
        
        self.income_keywords = ['sale', 'payment', 'deposit', 'transfer in', 'revenue']
    
    def process_bank_statement_pdf(self, file_content: bytes) -> List[ProcessedTransaction]:
        """Process PDF bank statement and extract transactions"""
        transactions = []
        
        try:
            pdf_reader = PyPDF2.PdfReader(file_content)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Parse transactions using regex patterns
            # Pattern for common bank statement formats: DATE DESCRIPTION AMOUNT BALANCE
            pattern = r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(.+?)\s+([\-\+]?\d+\.?\d*)\s+(\d+\.?\d*)'
            matches = re.findall(pattern, text, re.MULTILINE)
            
            for match in matches:
                date_str, description, amount_str, balance_str = match
                
                try:
                    date = datetime.strptime(date_str, "%m/%d/%Y")
                    amount = float(amount_str.replace(',', ''))
                    balance = float(balance_str.replace(',', ''))
                    
                    # Categorize transaction
                    category, subcategory = self._categorize_transaction(description, amount)
                    transaction_type = TransactionCategory.INCOME if amount > 0 else TransactionCategory.EXPENSE
                    
                    transactions.append(ProcessedTransaction(
                        date=date,
                        amount=amount,
                        description=description.strip(),
                        category=category,
                        subcategory=subcategory,
                        transaction_type=transaction_type,
                        balance_after=balance
                    ))
                except ValueError:
                    continue
                    
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
        
        return sorted(transactions, key=lambda x: x.date)
    
    def process_excel_statements(self, file_content: bytes) -> List[ProcessedTransaction]:
        """Process Excel bank statements"""
        transactions = []
        
        try:
            df = pd.read_excel(file_content)
            
            # Common column mappings
            column_mappings = {
                'date': ['date', 'transaction date', 'posting date'],
                'description': ['description', 'memo', 'details', 'reference'],
                'amount': ['amount', 'debit', 'credit', 'transaction amount'],
                'balance': ['balance', 'running balance', 'available balance']
            }
            
            # Map columns to standard format
            mapped_cols = {}
            for std_col, possible_names in column_mappings.items():
                for col in df.columns:
                    if col.lower().strip() in possible_names:
                        mapped_cols[std_col] = col
                        break
            
            for _, row in df.iterrows():
                try:
                    date = pd.to_datetime(row[mapped_cols['date']])
                    amount = float(str(row[mapped_cols['amount']]).replace(',', ''))
                    description = str(row[mapped_cols['description']])
                    balance = float(str(row[mapped_cols.get('balance', 0)]).replace(',', ''))
                    
                    category, subcategory = self._categorize_transaction(description, amount)
                    transaction_type = TransactionCategory.INCOME if amount > 0 else TransactionCategory.EXPENSE
                    
                    transactions.append(ProcessedTransaction(
                        date=date,
                        amount=amount,
                        description=description,
                        category=category,
                        subcategory=subcategory,
                        transaction_type=transaction_type,
                        balance_after=balance
                    ))
                except (ValueError, KeyError):
                    continue
                    
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing Excel: {str(e)}")
        
        return sorted(transactions, key=lambda x: x.date)
    
    def _categorize_transaction(self, description: str, amount: float) -> tuple:
        """Categorize transaction based on description and amount"""
        desc_lower = description.lower()
        
        if amount > 0:
            return "income", "sales"
        
        # Check expense categories
        for category, keywords in self.expense_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                return "expense", category
        
        return "expense", "other"

class FeatureEngineer:
    """Generate features for ML models"""
    
    def create_cashflow_features(self, transactions: List[ProcessedTransaction]) -> pd.DataFrame:
        """Create features for cash flow prediction"""
        df = pd.DataFrame([{
            'date': t.date,
            'amount': t.amount,
            'category': t.category,
            'balance': t.balance_after
        } for t in transactions])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        # Rolling statistics
        df['balance_ma_7'] = df['balance'].rolling(window=7, min_periods=1).mean()
        df['balance_ma_30'] = df['balance'].rolling(window=30, min_periods=1).mean()
        df['amount_ma_7'] = df['amount'].rolling(window=7, min_periods=1).mean()
        df['amount_std_7'] = df['amount'].rolling(window=7, min_periods=1).std()
        
        # Cash flow metrics
        df['daily_inflow'] = df[df['amount'] > 0]['amount'].rolling(window=7, min_periods=1).sum()
        df['daily_outflow'] = df[df['amount'] < 0]['amount'].rolling(window=7, min_periods=1).sum()
        df['net_flow'] = df['daily_inflow'] + df['daily_outflow']  # outflow is negative
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df
    
    def create_operational_features(self, transactions: List[ProcessedTransaction]) -> Dict:
        """Create operational intelligence features"""
        df = self.create_cashflow_features(transactions)
        
        # Calculate key metrics
        total_revenue = df[df['amount'] > 0]['amount'].sum()
        total_expenses = abs(df[df['amount'] < 0]['amount'].sum())
        avg_daily_balance = df['balance'].mean()
        balance_volatility = df['balance'].std()
        
        # Payment patterns
        income_transactions = df[df['amount'] > 0]
        avg_payment_size = income_transactions['amount'].mean() if len(income_transactions) > 0 else 0
        payment_frequency = len(income_transactions) / len(df) if len(df) > 0 else 0
        
        # Expense analysis
        expense_by_category = df[df['amount'] < 0].groupby('category')['amount'].sum().abs()
        
        return {
            'total_revenue': total_revenue,
            'total_expenses': total_expenses,
            'profit_margin': (total_revenue - total_expenses) / total_revenue if total_revenue > 0 else 0,
            'avg_daily_balance': avg_daily_balance,
            'balance_volatility': balance_volatility,
            'avg_payment_size': avg_payment_size,
            'payment_frequency': payment_frequency,
            'expense_breakdown': expense_by_category.to_dict(),
            'cash_cycle_days': self._calculate_cash_cycle(df)
        }
    
    def _calculate_cash_cycle(self, df: pd.DataFrame) -> float:
        """Calculate average cash conversion cycle"""
        positive_flows = df[df['amount'] > 0]['amount']
        negative_flows = df[df['amount'] < 0]['amount']
        
        if len(positive_flows) == 0 or len(negative_flows) == 0:
            return 0
        
        avg_positive = positive_flows.mean()
        avg_negative = abs(negative_flows.mean())
        
        # Simplified cash cycle calculation
        return avg_positive / avg_negative if avg_negative > 0 else 0

# FastAPI Application
app = FastAPI(title="FinanceOS Data Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
DATABASE_URL = "postgresql://user:password@localhost/financeos"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Redis setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize processors
doc_processor = DocumentProcessor()
feature_engineer = FeatureEngineer()

@app.post("/api/upload/bank-statement")
async def upload_bank_statement(
    file: UploadFile = File(...),
    user_id: str = "default",
    db: Session = Depends(get_db)
):
    """Upload and process bank statement"""
    
    if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Only PDF and Excel files supported")
    
    content = await file.read()
    
    # Process based on file type
    if file.content_type == "application/pdf":
        transactions = doc_processor.process_bank_statement_pdf(content)
    else:
        transactions = doc_processor.process_excel_statements(content)
    
    # Save to database
    db_transactions = []
    for t in transactions:
        db_transaction = Transaction(
            user_id=user_id,
            date=t.date,
            amount=t.amount,
            description=t.description,
            category=t.category,
            subcategory=t.subcategory,
            transaction_type=t.transaction_type.value,
            balance_after=t.balance_after
        )
        db_transactions.append(db_transaction)
    
    db.add_all(db_transactions)
    db.commit()
    
    # Generate features and cache
    features = feature_engineer.create_operational_features(transactions)
    redis_client.setex(f"features:{user_id}", 3600, json.dumps(features, default=str))
    
    return {
        "message": f"Successfully processed {len(transactions)} transactions",
        "transactions_count": len(transactions),
        "date_range": {
            "start": min(t.date for t in transactions).isoformat(),
            "end": max(t.date for t in transactions).isoformat()
        },
        "summary": features
    }

@app.get("/api/transactions/{user_id}")
def get_transactions(user_id: str, db: Session = Depends(get_db)):
    """Get user transactions"""
    transactions = db.query(Transaction).filter(Transaction.user_id == user_id).all()
    return transactions

@app.get("/api/features/{user_id}")
def get_features(user_id: str):
    """Get cached features for user"""
    cached_features = redis_client.get(f"features:{user_id}")
    if cached_features:
        return json.loads(cached_features)
    return {"error": "No features found"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
