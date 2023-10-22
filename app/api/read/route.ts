import { NextRequest, NextResponse } from 'next/server'
import { PineconeClient } from '@pinecone-database/pinecone'
import {
  queryPineconeVectorStoreAndQueryLLM,
} from '../../../utils'
import { indexName } from '../../../config'

export async function POST(req: NextRequest) {
  const body = await req.json()
  const client = new PineconeClient()
  await client.init({
    apiKey: process.env.PINECONE_API_KEY || '',
    environment: process.env.PINECONE_ENVIRONMENT || ''
  })

  const result = await queryPineconeVectorStoreAndQueryLLM(client, indexName, body)
  if (!result) {
    // Handle the 'undefined' case here, perhaps by returning a default value or throwing an error
    throw new Error("Received undefined result from queryPineconeVectorStoreAndQueryLLM");
  }
  
  const [text, context] = result; 
  return NextResponse.json({
    data: text, context
  })
}